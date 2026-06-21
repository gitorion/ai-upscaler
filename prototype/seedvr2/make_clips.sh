#!/bin/bash
##############################################################################
# SeedVR2 prototype — cut representative clips for evaluation
# Usage: ./make_clips.sh <source_video> <start1> [start2] [start3] ...
#   starts are HH:MM:SS (or seconds). Each clip is CLIP_LEN seconds (default 8).
# Clips are stream-copied (lossless, instant) into the runtime eval dir.
# Pick moments that stress the model: faces, fine texture, fast motion,
# heavy compression / dark scenes.
##############################################################################
set -eo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SELF_DIR/_env.sh"

CLIP_LEN="${CLIP_LEN:-8}"   # seconds per clip; override: CLIP_LEN=12 ./make_clips.sh ...

if [[ $# -lt 2 ]]; then
    err "Usage: $0 <source_video> <start1> [start2] ..."
    echo "       e.g. $0 source.mkv 00:05:30 00:21:00 00:48:15"
    echo "       (CLIP_LEN=$CLIP_LEN s each; override with CLIP_LEN=N)"
    exit 1
fi

SRC="$1"; shift
[[ -f "$SRC" ]] || { err "Source not found: $SRC"; exit 1; }
command -v ffmpeg >/dev/null || { err "ffmpeg not found"; exit 1; }

mkdir -p "$CLIPS_DIR"

i=0
for start in "$@"; do
    i=$((i+1))
    out=$(printf "%s/clip_%02d.mkv" "$CLIPS_DIR" "$i")
    info "Clip $i: start=$start len=${CLIP_LEN}s → $out"
    # -ss before -i = fast seek; stream copy keeps it lossless & instant.
    # If a clip glitches at the very start (mid-GOP seek), re-encode that one:
    # swap '-c copy' for '-c:v libx264 -crf 12 -c:a copy'.
    ffmpeg -y -loglevel error -ss "$start" -i "$SRC" -t "$CLIP_LEN" -c copy "$out" \
        || err "Clip $i failed — try re-encoding (see comment in script)."
done

ok "Done. $(ls "$CLIPS_DIR"/clip_*.mkv 2>/dev/null | wc -l) clip(s) in $CLIPS_DIR/"
info "Next: ./run.sh $CLIPS_DIR/clip_01.mkv 1080"
