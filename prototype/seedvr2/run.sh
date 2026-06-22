#!/bin/bash
##############################################################################
# SeedVR2 prototype — upscale one clip with 16GB-safe settings
# Usage: ./run.sh <input_video> [target_short_edge_px]
#   e.g. ./run.sh ~/ai-upscale/seedvr2/eval/clips/clip_01.mkv 1080
# Output: <runtime>/seedvr2/eval/out/<name>_seedvr2.mkv  (HEVC 10-bit)
#
# ⚠️ The upstream SeedVR2 CLI changes flag names occasionally. If this errors
#    with "unrecognized arguments", run `python <CLI> -h` (path printed below)
#    and adjust the variable block / arg array to match.
##############################################################################
set -eo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SELF_DIR/_env.sh"

# ── 16GB tuning knobs — edit to taste ────────────────────────────────────────
# 3B FP8 is the right first choice for a 4060 Ti 16GB. Confirm the exact name +
# flag with `python "$CLI" -h` — newer builds use --dit_model, older --model.
MODEL_FILE="seedvr2_ema_3b_fp8_e4m3fn.safetensors"
MODEL_FLAG="--dit_model"     # change to "--model" if -h says so
BATCH_SIZE=13                # 4n+1 (1,5,9,13,17,33...). Higher = better temporal consistency
                             # AND faster, but more VRAM. Ideally matches shot length. On 16GB
                             # try 13→17→33; drop toward 5/1 if you OOM. 5 looks needlessly flickery.
BLOCKS_TO_SWAP=16            # raise to 24–32 if OOM; lower for speed
COLOR_CORRECTION="lab"       # lab | wavelet | adain | none
TEMPORAL_OVERLAP=3           # frames blended between batches — smooths seams (examples use 3)
# VAE tile sizes (px) — tuned separately: encode fit fine at 1024, only decode OOM'd.
# Larger tile = fewer tiles = faster, up to the VRAM limit. Decode is the bottleneck.
VAE_ENCODE_TILE_SIZE=1024    # fit comfortably at 1024 — keep it big for speed
VAE_DECODE_TILE_SIZE=768     # 1024 OOMs, 512 is slow (~15 tiles). 768 is the middle ground.
                             # If 768 OOMs in decode, drop to 640 then 512.
# Streaming chunk size (frames). Loading a whole clip holds the ENTIRE output + latents
# in RAM, which scales with length and OOM-kills on long clips (a 57s/1439-frame clip
# needed >32GB). Chunking keeps RAM flat regardless of length; temporal_overlap blends
# the seams. 0 = load all (short clips only). 250 is safe on 32GB; raise for fewer seams
# if RAM allows, lower toward 130 if you still OOM.
CHUNK_SIZE=250

# Lossless speedups (identical output, just faster). torch.compile fuses kernels via Triton
# (already installed). ATTENTION 'auto' uses flash_attn_2 if flash-attn is installed in the venv
# (lossless + faster), else falls back to 'sdpa' — so you only `pip install flash-attn`, no flag
# to flip. Pin to 'sdpa'/'flash_attn_2' to override. Do NOT use sageattn_* — QUANTIZED (lossy).
COMPILE=true
ATTENTION="auto"

# Post-process: SeedVR2 drops audio and has no denoise, so we add both in one ffmpeg
# pass after inference. DENOISE_VF is a minimal grain-reducer applied to the upscaled
# 1080p output. Set DENOISE_VF="" to disable (then video is stream-copied, lossless).
#   minimal (default): hqdn3d=1:1:2:2   |   main-pipeline 'light': hqdn3d=1.5:1:1.5:1
DENOISE_VF="hqdn3d=1:1:2:2"
# ─────────────────────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    err "Usage: $0 <input_video> [target_short_edge_px]"
    exit 1
fi
INPUT="$1"
RESOLUTION="${2:-1080}"
[[ -f "$INPUT" ]] || { err "Input not found: $INPUT"; exit 1; }
[[ -f "$CLI"   ]] || { err "SeedVR2 CLI not found at $CLI — run ./setup.sh first"; exit 1; }

mkdir -p "$OUT_DIR"
base=$(basename "$INPUT"); base="${base%.*}"
OUTPUT="$OUT_DIR/${base}_seedvr2.mkv"          # final (with audio + optional denoise)
RAW_OUTPUT="$OUT_DIR/${base}_seedvr2.raw.mkv"  # SeedVR2 video-only intermediate
DEANAMORPH="$OUT_DIR/${base}_square.mkv"        # de-anamorphized feed for SeedVR2 (if needed)

# ── Pre-process: de-anamorphize so SeedVR2 works at correct geometry ──────────────
# SeedVR2 has no aspect awareness — it upscales raw stored pixels. For anamorphic
# sources (e.g. PAL 720x576 shown 4:3) that yields a squished result. We fix the
# SHAPE BEFORE the AI so it reconstructs detail at correct proportions, instead of
# stretching (and softening) afterwards. Square-pixel sources pass through untouched.
# Lossless x264 (-qp 0) keeps the only pre-AI step from adding compression artifacts.
UPSCALE_INPUT="$INPUT"

read -r src_w src_h < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
          -of csv=p=0:nk=1 "$INPUT" 2>/dev/null | head -1 | tr ',' ' ')
src_dar=$(ffprobe -v error -select_streams v:0 -show_entries stream=display_aspect_ratio \
          -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1)
src_sar=$(ffprobe -v error -select_streams v:0 -show_entries stream=sample_aspect_ratio \
          -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1)

# Display width:height ratio (dar_w:dar_h): DAR → SAR×storage → storage pixels.
if [[ "$src_dar" =~ ^[0-9]+:[0-9]+$ && "$src_dar" != "0:1" ]]; then
    dar_w=${src_dar%:*}; dar_h=${src_dar#*:}
elif [[ "$src_sar" =~ ^[0-9]+:[0-9]+$ && "$src_sar" != "0:1" && -n "$src_w" && -n "$src_h" ]]; then
    sar_w=${src_sar%:*}; sar_h=${src_sar#*:}; dar_w=$(( src_w * sar_w )); dar_h=$(( src_h * sar_h ))
else
    dar_w=${src_w:-0}; dar_h=${src_h:-0}
fi

# Square-pixel width at native height. If it differs from stored width → anamorphic.
if [[ "${src_h:-0}" -gt 0 && "${dar_h:-0}" -gt 0 ]]; then
    square_w=$(awk -v h="$src_h" -v n="$dar_w" -v d="$dar_h" \
               'BEGIN{w=int(h*n/d + 0.5); if(w%2)w--; print w}')
    if [[ "$square_w" -gt 0 && "$square_w" != "$src_w" ]]; then
        info "De-anamorphize: ${src_w}x${src_h} (SAR ${src_sar:-?}) → ${square_w}x${src_h} square px (display ${dar_w}:${dar_h})"
        if ffmpeg -y -loglevel error -i "$INPUT" \
               -vf "scale=${square_w}:${src_h}:flags=lanczos,setsar=1" \
               -an -sn -c:v libx264 -qp 0 -pix_fmt yuv420p "$DEANAMORPH"; then
            UPSCALE_INPUT="$DEANAMORPH"
        else
            warn "De-anamorphize failed — feeding original; post-process will correct AR instead."
        fi
    else
        info "Source is square-pixel (${src_w}x${src_h}) — no de-anamorphize needed"
    fi
fi

# shellcheck disable=SC1091
source "$SEEDVR2_VENV/bin/activate"

info "SeedVR2 → $OUTPUT"
info "model=$MODEL_FILE  res=${RESOLUTION}  batch=${BATCH_SIZE}  blocks_swap=${BLOCKS_TO_SWAP}"
warn "First run downloads weights to $MODEL_DIR (~several GB) — be patient."

# Resolve attention 'auto' → flash_attn_2 if flash-attn is importable in the venv, else sdpa.
attn="$ATTENTION"
if [[ "$attn" == "auto" ]]; then
    if python -c "import flash_attn" &>/dev/null; then
        attn="flash_attn_2"; info "flash-attn detected — using flash_attn_2 (lossless speedup)"
    else
        attn="sdpa"
    fi
fi

# Args as an array so flags are easy to add/remove.
args=(
    "$UPSCALE_INPUT"             # de-anamorphized feed when source is anamorphic, else original
    --output "$RAW_OUTPUT"       # video-only; audio is remuxed in the post-process step
    --output_format mp4          # stream encoded by --video_backend; .mkv name is fine
    --video_backend ffmpeg
    --10bit                      # HEVC 10-bit, matching the main pipeline's quality target
    --model_dir "$MODEL_DIR"
    "$MODEL_FLAG" "$MODEL_FILE"
    --resolution "$RESOLUTION"
    --batch_size "$BATCH_SIZE"
    --chunk_size "$CHUNK_SIZE"   # streaming: keeps system-RAM flat vs clip length (avoids OOM-kill)
    --temporal_overlap "$TEMPORAL_OVERLAP"
    --color_correction "$COLOR_CORRECTION"
    --blocks_to_swap "$BLOCKS_TO_SWAP"
    --dit_offload_device cpu
    --vae_offload_device cpu     # free the VAE from VRAM between phases (helps decode headroom)
    --vae_encode_tiled
    --vae_encode_tile_size "$VAE_ENCODE_TILE_SIZE"
    --vae_decode_tiled
    --vae_decode_tile_size "$VAE_DECODE_TILE_SIZE"
    --attention_mode "$attn"        # resolved above; lossless (flash_attn_2 faster if installed)
)
[[ "$COMPILE" == true ]] && args+=(--compile_dit --compile_vae)   # lossless torch.compile speedup

info "Command: python $CLI ${args[*]}"
echo ""

SECONDS=0
python "$CLI" "${args[@]}"
elapsed=$SECONDS

[[ -f "$RAW_OUTPUT" ]] || { err "SeedVR2 produced no output ($RAW_OUTPUT) — see errors above."; exit 1; }

# ── Post-process: AR safety-net + optional denoise + remux audio ──────────────────
# Aspect is normally fixed up front (de-anamorphize). This stays as a SAFETY NET: if
# the pre-process was skipped/failed, we still correct the shape here using the source
# DAR (dar_w:dar_h, already resolved above). Normally raw == target, so this no-ops and
# the video is stream-copied. We also remux audio/subs from the ORIGINAL input (frame
# count + fps are preserved, so it stays in sync; '?' makes those maps optional).

has_audio=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_type \
            -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null || true)

# SeedVR2 output dimensions (height is the short edge it produced).
read -r raw_w raw_h < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
          -of csv=p=0:nk=1 "$RAW_OUTPUT" 2>/dev/null | head -1 | tr ',' ' ')

# Correct width for the height SeedVR2 produced (even). Guard against bad probe.
target_h=${raw_h:-0}
if [[ "$target_h" -gt 0 && "$dar_h" -gt 0 ]]; then
    target_w=$(awk -v h="$target_h" -v n="$dar_w" -v d="$dar_h" \
               'BEGIN{w=int(h*n/d + 0.5); if(w%2)w--; print w}')
else
    target_w=${raw_w:-0}
fi

# Build the video filter chain: scale (only if AR correction is needed) + denoise.
vf_parts=()
need_reencode=false
if [[ "$target_w" -gt 0 && ( "$target_w" != "$raw_w" || "$target_h" != "$raw_h" ) ]]; then
    vf_parts+=("scale=${target_w}:${target_h}:flags=lanczos")
    need_reencode=true
    info "Aspect: ${raw_w}x${raw_h} → ${target_w}x${target_h} (display AR ${dar_w}:${dar_h})"
else
    info "Aspect: ${raw_w}x${raw_h} already correct — no rescale"
fi
[[ -n "$DENOISE_VF" ]] && { vf_parts+=("$DENOISE_VF"); need_reencode=true; }

if [[ "$need_reencode" == true ]]; then
    vf_chain=$(IFS=,; echo "${vf_parts[*]}")
    info "Post-process: re-encode HEVC 10-bit (vf: $vf_chain) + remux audio"
    video_opts=(-vf "$vf_chain" -c:v libx265 -crf 16 -preset medium
                -pix_fmt yuv420p10le -x265-params "aq-mode=3")
else
    info "Post-process: remux audio only → stream-copying video (lossless)"
    video_opts=(-c:v copy)
fi

ffmpeg -y -loglevel error -i "$RAW_OUTPUT" -i "$INPUT" \
    -map 0:v:0 -map "1:a?" -map "1:s?" \
    "${video_opts[@]}" -c:a copy -c:s copy -shortest \
    "$OUTPUT" \
    && rm -f "$RAW_OUTPUT" \
    || { err "Post-process failed — raw video-only result kept at $RAW_OUTPUT"; exit 1; }

rm -f "$DEANAMORPH"   # drop the de-anamorphized intermediate (no-op if unused)

[[ -n "$has_audio" ]] && ok "Audio retained." || warn "Source had no audio stream — output is silent."

echo ""
ok "Done in ${elapsed}s (+ post-process) → $OUTPUT"
info "A/B it: $MAIN_UPSCALER -i \"$INPUT\" -r ${RESOLUTION}p -m basicvsr -o \"$OUT_DIR/${base}_basicvsr.mkv\""
info "Then:   ./compare.sh \"$OUT_DIR/${base}_basicvsr.mkv\" \"$OUTPUT\""
