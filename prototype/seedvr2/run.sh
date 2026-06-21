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
OUTPUT="$OUT_DIR/${base}_seedvr2.mkv"

# shellcheck disable=SC1091
source "$SEEDVR2_VENV/bin/activate"

info "SeedVR2 → $OUTPUT"
info "model=$MODEL_FILE  res=${RESOLUTION}  batch=${BATCH_SIZE}  blocks_swap=${BLOCKS_TO_SWAP}"
warn "First run downloads weights to $MODEL_DIR (~several GB) — be patient."

# Args as an array so flags are easy to add/remove.
args=(
    "$INPUT"
    --output "$OUTPUT"
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
    --vae_encode_tiled
    --vae_decode_tiled
)

info "Command: python $CLI ${args[*]}"
echo ""

SECONDS=0
python "$CLI" "${args[@]}"
elapsed=$SECONDS

echo ""
ok "Done in ${elapsed}s → $OUTPUT"
info "A/B it: $MAIN_UPSCALER -i \"$INPUT\" -r ${RESOLUTION}p -m basicvsr -o \"$OUT_DIR/${base}_basicvsr.mkv\""
info "Then:   ./compare.sh \"$OUT_DIR/${base}_basicvsr.mkv\" \"$OUTPUT\""
