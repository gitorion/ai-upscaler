#!/bin/bash
##############################################################################
# SeedVR2 A/B test — compare prefilter/temporal-overlap settings on a short clip
# Usage: ./abtest.sh <source_video> <start HH:MM:SS>
#   Cuts a CLIP_LEN-second clip at <start>, upscales it 3 ways with -m seedvr2,
#   and stacks the results side by side so you can judge a specific artifact
#   (e.g. ghosting/noise on a static wall during motion).
#
#   Variants:
#     current   — prefilter none (seedvr2 default), temporal_overlap 3 (default)
#     pf_light  — prefilter light (cleaner input — targets the "noisy layer")
#     overlap0  — temporal_overlap 0 (no batch-seam blend — targets the "ghost layer")
#
# Runs in an ISOLATED temp dir (UPSCALE_TEMP_DIR) so it does NOT touch / wipe a
# paused production run's segments in ~/ai-upscale/temp. Resumable: variants whose
# output already exists are skipped, so a re-run continues where it left off.
##############################################################################
set -eo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SELF_DIR/_env.sh"

MAIN_UPSCALER="$(cd "$SELF_DIR/../.." && pwd)/upscale_video.sh"
AB_TEMP="$RUNTIME_DIR/temp_abtest"          # isolated temp — keeps production temp safe
AB_DIR="$EVAL_DIR/abtest"
CLIP_LEN="${CLIP_LEN:-45}"                  # seconds; lower (e.g. 30/20) to speed the test up
RES="${RES:-1080p}"

if [[ $# -lt 2 ]]; then
    err "Usage: $0 <source_video> <start HH:MM:SS>"
    echo "   e.g. $0 ~/desperation-480p.mkv 00:14:30   (pick the wall-motion shot)"
    echo "   CLIP_LEN=$CLIP_LEN s (override: CLIP_LEN=30 $0 ...), RES=$RES"
    exit 1
fi
SRC="$1"; START="$2"
[[ -f "$SRC" ]]           || { err "Source not found: $SRC"; exit 1; }
[[ -f "$MAIN_UPSCALER" ]] || { err "Upscaler not found: $MAIN_UPSCALER"; exit 1; }
command -v ffmpeg >/dev/null || { err "ffmpeg not found"; exit 1; }

mkdir -p "$AB_DIR"
CLIP="$AB_DIR/clip.mkv"

# Cut the test clip (lossless stream copy).
if [[ -f "$CLIP" ]]; then
    info "Reusing existing clip: $CLIP"
else
    info "Cutting ${CLIP_LEN}s clip at ${START}..."
    ffmpeg -y -loglevel error -ss "$START" -i "$SRC" -t "$CLIP_LEN" -c copy "$CLIP"
fi

# ── Run one variant (skip if already done) ───────────────────────────────────
# Args: name  temporal_overlap  [extra upscaler flags...]
run_variant() {
    local name="$1" overlap="$2"; shift 2
    local out="$AB_DIR/${name}.mkv"
    if [[ -f "$out" ]]; then
        ok "Variant '${name}' already done — skipping"
        return
    fi
    info "── Variant '${name}': overlap=${overlap} ${*:+extra: $*} ──"
    SECONDS=0
    UPSCALE_TEMP_DIR="$AB_TEMP" SEEDVR2_TEMPORAL_OVERLAP="$overlap" \
        "$MAIN_UPSCALER" -i "$CLIP" -r "$RES" -m seedvr2 -o "$out" "$@"
    ok "Variant '${name}' done in ${SECONDS}s → $out"
}

warn "Each variant upscales the full clip (~minutes/frame). 3 variants on a ${CLIP_LEN}s clip"
warn "will take a few hours total. Make sure the production run is paused (it shares the GPU)."

run_variant "current"  3                       # prefilter none (default), overlap 3
run_variant "pf_light" 3  --prefilter light    # cleaner input
run_variant "overlap0" 0                        # no batch-seam blend

# ── Stack the three side by side for comparison ──────────────────────────────
# Scale each to 720h honoring display AR; label if a font is available.
FONT=""
for f in /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf /usr/share/fonts/TTF/DejaVuSans.ttf \
         /usr/share/fonts/dejavu/DejaVuSans.ttf; do
    [[ -f "$f" ]] && { FONT="$f"; break; }
done
lbl() { [[ -n "$FONT" ]] && printf ",drawtext=fontfile='%s':text='%s':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5" "$FONT" "$1"; }

COMPARE="$AB_DIR/abtest_compare.mkv"
info "Stacking variants → $COMPARE"
ffmpeg -y -loglevel error \
    -i "$AB_DIR/current.mkv" -i "$AB_DIR/pf_light.mkv" -i "$AB_DIR/overlap0.mkv" \
    -filter_complex "\
        [0:v]scale=w=trunc(720*dar/2)*2:h=720,setsar=1$(lbl 'current (none, ov3)')[a];\
        [1:v]scale=w=trunc(720*dar/2)*2:h=720,setsar=1$(lbl 'pf=light')[b];\
        [2:v]scale=w=trunc(720*dar/2)*2:h=720,setsar=1$(lbl 'overlap=0')[c];\
        [a][b][c]hstack=inputs=3" \
    -c:v libx265 -crf 18 -preset medium -pix_fmt yuv420p10le -an "$COMPARE" \
    || { err "Stacking failed (often a missing font). The three variant files are still in $AB_DIR."; exit 1; }

rm -rf "$AB_TEMP" 2>/dev/null || true
ok "Done. Compare: $COMPARE"
info "Individual outputs: $AB_DIR/{current,pf_light,overlap0}.mkv"
info "Watch the static wall during motion: which variant has the least ghost/noise wins."
