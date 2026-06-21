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

# shellcheck disable=SC1091
source "$SEEDVR2_VENV/bin/activate"

info "SeedVR2 → $OUTPUT"
info "model=$MODEL_FILE  res=${RESOLUTION}  batch=${BATCH_SIZE}  blocks_swap=${BLOCKS_TO_SWAP}"
warn "First run downloads weights to $MODEL_DIR (~several GB) — be patient."

# Args as an array so flags are easy to add/remove.
args=(
    "$INPUT"
    --output "$RAW_OUTPUT"       # video-only; audio is remuxed in the post-process step
    --output_format mp4          # stream encoded by --video_backend; .mkv name is fine
    --video_backend ffmpeg
    --10bit                      # HEVC 10-bit, matching the main pipeline's quality target
    --model_dir "$MODEL_DIR"
    "$MODEL_FLAG" "$MODEL_FILE"
    --resolution "$RESOLUTION"
    --batch_size "$BATCH_SIZE"
    --temporal_overlap "$TEMPORAL_OVERLAP"
    --color_correction "$COLOR_CORRECTION"
    --blocks_to_swap "$BLOCKS_TO_SWAP"
    --dit_offload_device cpu
    --vae_offload_device cpu     # free the VAE from VRAM between phases (helps decode headroom)
    --vae_encode_tiled
    --vae_encode_tile_size "$VAE_ENCODE_TILE_SIZE"
    --vae_decode_tiled
    --vae_decode_tile_size "$VAE_DECODE_TILE_SIZE"
)

info "Command: python $CLI ${args[*]}"
echo ""

SECONDS=0
python "$CLI" "${args[@]}"
elapsed=$SECONDS

[[ -f "$RAW_OUTPUT" ]] || { err "SeedVR2 produced no output ($RAW_OUTPUT) — see errors above."; exit 1; }

# ── Post-process: remux original audio (+ subs) and optional minimal denoise ──────
# SeedVR2 outputs video only. We pull audio/subs from the ORIGINAL input. Frame count
# and fps are preserved by SeedVR2, so a straight remux stays in sync. The '?' on the
# audio/subtitle maps makes them optional (no failure if the source has none).
has_audio=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_type \
            -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null || true)

if [[ -n "$DENOISE_VF" ]]; then
    info "Post-process: denoise ($DENOISE_VF) + remux audio → re-encoding HEVC 10-bit"
    video_opts=(-vf "$DENOISE_VF" -c:v libx265 -crf 16 -preset medium
                -pix_fmt yuv420p10le -x265-params "aq-mode=3")
else
    info "Post-process: remux audio (no denoise) → stream-copying video (lossless)"
    video_opts=(-c:v copy)
fi

ffmpeg -y -loglevel error -i "$RAW_OUTPUT" -i "$INPUT" \
    -map 0:v:0 -map "1:a?" -map "1:s?" \
    "${video_opts[@]}" -c:a copy -c:s copy -shortest \
    "$OUTPUT" \
    && rm -f "$RAW_OUTPUT" \
    || { err "Post-process failed — raw video-only result kept at $RAW_OUTPUT"; exit 1; }

[[ -n "$has_audio" ]] && ok "Audio retained." || warn "Source had no audio stream — output is silent."

echo ""
ok "Done in ${elapsed}s (+ post-process) → $OUTPUT"
info "A/B it: $MAIN_UPSCALER -i \"$INPUT\" -r ${RESOLUTION}p -m basicvsr -o \"$OUT_DIR/${base}_basicvsr.mkv\""
info "Then:   ./compare.sh \"$OUT_DIR/${base}_basicvsr.mkv\" \"$OUTPUT\""
