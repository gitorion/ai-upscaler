#!/bin/bash
##############################################################################
# SeedVR2 prototype — side-by-side comparison
# Usage: ./compare.sh <left.mkv> <right.mkv> [output.mkv]
#   e.g. ./compare.sh out/clip_01_basicvsr.mkv out/clip_01_seedvr2.mkv
# Stacks both videos horizontally (scaled to equal height) with labels so you
# can eyeball detail, hallucination, and temporal stability frame by frame.
# Output defaults to the runtime eval dir.
##############################################################################
set -eo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SELF_DIR/_env.sh"

if [[ $# -lt 2 ]]; then
    err "Usage: $0 <left.mkv> <right.mkv> [output.mkv]"
    exit 1
fi
LEFT="$1"; RIGHT="$2"; OUT="${3:-$OUT_DIR/compare.mkv}"
[[ -f "$LEFT"  ]] || { err "Not found: $LEFT";  exit 1; }
[[ -f "$RIGHT" ]] || { err "Not found: $RIGHT"; exit 1; }
command -v ffmpeg >/dev/null || { err "ffmpeg not found"; exit 1; }
mkdir -p "$(dirname "$OUT")"

LEFT_LABEL="${LEFT_LABEL:-$(basename "$LEFT")}"
RIGHT_LABEL="${RIGHT_LABEL:-$(basename "$RIGHT")}"

# Scale both to 1080 height, label each, hstack. drawtext needs a font; if
# fontconfig has no default the labels just won't render (video still stacks).
info "Stacking: [$LEFT_LABEL] | [$RIGHT_LABEL] → $OUT"
ffmpeg -y -loglevel error \
    -i "$LEFT" -i "$RIGHT" \
    -filter_complex "\
        [0:v]scale=-2:1080,setsar=1,drawtext=text='${LEFT_LABEL}':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[l];\
        [1:v]scale=-2:1080,setsar=1,drawtext=text='${RIGHT_LABEL}':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[r];\
        [l][r]hstack=inputs=2" \
    -c:v libx265 -crf 18 -preset medium -pix_fmt yuv420p10le \
    -an "$OUT" \
    || { err "Comparison failed (often a missing font for drawtext)."; exit 1; }

ok "Wrote $OUT — scrub through it to judge detail / hallucination / flicker."
