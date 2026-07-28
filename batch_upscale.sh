#!/bin/bash

##############################################################################
# Batch AI Video Upscaler — run upscale_video.sh over a whole folder (a season)
#
# Processes every video in an input folder, one at a time, producing one finished
# file per episode. Designed for long unattended runs (e.g. 9 episodes of SeedVR2):
#
#   * Resumable at TWO levels:
#       - between episodes: an episode whose output already exists & is valid is
#         SKIPPED, so re-running the same command after 3 finished episodes just
#         picks up at episode 4. No state file — the finished outputs ARE the state.
#       - within an episode: the child run is always invoked with --resume, so an
#         episode interrupted mid-way (its temp/segments survive) continues where
#         it stopped rather than restarting.
#   * Low disk usage: the child cleans up its own temp when an episode finishes, so
#     only ONE episode's working files exist at a time plus the completed outputs.
#   * Quality-first: every setting you pass (-m, --prefilter, -q, ...) is forwarded
#     verbatim to upscale_video.sh — this wrapper adds no re-encode of its own.
#
# Usage:
#   ./batch_upscale.sh -i <folder> -r <resolution> [-o <out_folder>] [batch opts] [upscaler opts...]
#
#   e.g.  ./batch_upscale.sh -i ~/season_1 -r 1080p --prefilter none -m seedvr2
#
# Any option this script doesn't recognise is passed straight through to
# upscale_video.sh, so the full single-file flag set is available.
##############################################################################

set -uo pipefail   # NOT -e: we handle per-episode failures explicitly so one bad
                   # episode doesn't silently abort the batch without a report.

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[BATCH]${NC} $*"; }
ok()    { echo -e "${GREEN}[BATCH]${NC} $*"; }
warn()  { echo -e "${YELLOW}[BATCH]${NC} $*"; }
err()   { echo -e "${RED}[BATCH]${NC} $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSCALER="$SCRIPT_DIR/upscale_video.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
INPUT_DIR=""
OUTPUT_DIR=""
RESOLUTION=""
KEEP_GOING=false      # stop on first failed episode unless set
FORCE=false           # reprocess (OVERWRITE) episodes whose output already exists
PASSTHROUGH=()        # everything else → upscale_video.sh

# Video extensions we treat as episodes (case-insensitive).
VIDEO_EXTS="mkv mp4 avi mov m4v ts m2ts mts wmv flv webm mpg mpeg vob"

usage() {
    cat << EOF
${GREEN}Batch AI Video Upscaler${NC} — upscale a whole folder (a season) in one go

Usage: $0 -i FOLDER -r RESOLUTION [OPTIONS] [-- UPSCALER OPTIONS...]

Required:
  -i, --input FOLDER        Folder of episodes to upscale (non-recursive)
  -r, --resolution RES      Target resolution: 720p, 1080p, 1440p, 2160p

Batch options:
  -o, --output FOLDER       Where finished files go
                            (default: sibling folder FOLDER_RES, e.g. season_1_1080p)
  --keep-going              Continue to the next episode if one fails
                            (default: stop on the first failure and report)
  --force                   Reprocess episodes even if a finished output already
                            exists — ${YELLOW}this OVERWRITES those outputs${NC}
  -h, --help                Show this help

Any other option is forwarded to upscale_video.sh, e.g.:
  -m seedvr2   --prefilter none   -q high   --encode-speed slow   --deinterlace

Resuming:
  Just re-run the SAME command. Episodes with a valid finished output are skipped;
  an episode that was interrupted mid-way is resumed automatically. Nothing is lost.

Examples:
  # A season of 9 episodes with SeedVR2, no prefilter
  $0 -i ~/season_1 -r 1080p --prefilter none -m seedvr2

  # Same, but keep going past any episode that errors
  $0 -i ~/season_1 -r 1080p -m seedvr2 --keep-going

  # Custom output folder, standard (non-diffusion) model
  $0 -i ~/season_1 -o ~/season_1_4k -r 2160p -m nomos8k
EOF
    exit "${1:-0}"
}

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)       INPUT_DIR="$2";  shift 2 ;;
        -o|--output)      OUTPUT_DIR="$2"; shift 2 ;;
        -r|--resolution)  RESOLUTION="$2"; shift 2 ;;
        --keep-going)     KEEP_GOING=true; shift   ;;
        --force)          FORCE=true;      shift   ;;
        -h|--help)        usage 0 ;;
        --)               shift; PASSTHROUGH+=("$@"); break ;;
        *)                PASSTHROUGH+=("$1"); shift ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
[[ -x "$UPSCALER" || -f "$UPSCALER" ]] || { err "upscale_video.sh not found next to this script: $UPSCALER"; exit 1; }
command -v ffprobe >/dev/null 2>&1     || { err "ffprobe not found (install ffmpeg)"; exit 1; }

if [[ -z "$INPUT_DIR" ]]; then err "Missing input folder (-i)"; usage 1; fi
if [[ -z "$RESOLUTION" ]]; then err "Missing resolution (-r) — batch mode is non-interactive"; usage 1; fi
if [[ ! -d "$INPUT_DIR" ]]; then err "Input folder not found (must be a directory): $INPUT_DIR"; exit 1; fi

INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"                 # normalise to absolute
# Default output is a SIBLING folder named <input>_<resolution>, e.g. ~/season_1 → ~/season_1_1080p.
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$INPUT_DIR")/$(basename "$INPUT_DIR")_${RESOLUTION}}"
mkdir -p "$OUTPUT_DIR" || { err "Cannot create output folder: $OUTPUT_DIR"; exit 1; }
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# ── Enumerate episodes (non-recursive, natural sort) ──────────────────────────
# Build a case-insensitive -iname predicate list for find, then sort -V so
# ep2 < ep10. We deliberately exclude the output folder in case it sits inside INPUT_DIR.
find_args=()
first=true
for ext in $VIDEO_EXTS; do
    if $first; then find_args+=( -iname "*.$ext" ); first=false
    else            find_args+=( -o -iname "*.$ext" ); fi
done

# read-loop instead of mapfile so this works on bash 3.2 (macOS) as well as bash 5 (Ubuntu target)
EPISODES=()
while IFS= read -r _ep; do
    [[ -n "$_ep" ]] && EPISODES+=("$_ep")
done < <(
    find "$INPUT_DIR" -maxdepth 1 -type f \( "${find_args[@]}" \) \
        -not -path "$OUTPUT_DIR/*" 2>/dev/null | sort -V
)

if [[ ${#EPISODES[@]} -eq 0 ]]; then
    err "No video files found in $INPUT_DIR (looked for: $VIDEO_EXTS)"
    exit 1
fi

# ── Helper: is a file a valid, complete video? ────────────────────────────────
probe_ok() { ffprobe -v error -i "$1" -show_entries format=duration -of csv=p=0 >/dev/null 2>&1; }

# ── Plan summary ──────────────────────────────────────────────────────────────
info "=== Batch upscale ==="
info "Input folder:  $INPUT_DIR"
info "Output folder: $OUTPUT_DIR"
info "Resolution:    $RESOLUTION"
info "Episodes:      ${#EPISODES[@]}"
[[ ${#PASSTHROUGH[@]} -gt 0 ]] && info "Upscaler args: ${PASSTHROUGH[*]}"
$FORCE && warn "--force: existing finished outputs WILL be overwritten"
$KEEP_GOING && info "--keep-going: will continue past failed episodes"
echo ""

TOTAL=${#EPISODES[@]}
COUNT=0
DONE_SKIPPED=0
DONE_UPSCALED=0
FAILED=0
FAILED_LIST=()
BATCH_START=$SECONDS

for EP in "${EPISODES[@]}"; do
    COUNT=$((COUNT + 1))
    BASE="$(basename "$EP")"
    STEM="${BASE%.*}"
    OUT="$OUTPUT_DIR/${STEM}_upscaled_${RESOLUTION}.mkv"

    echo -e "${GREEN}────────────────────────────────────────────────────────────────${NC}"
    info "Episode $COUNT/$TOTAL: $BASE"

    # Already finished? (valid output present, and not forcing) → skip.
    if [[ "$FORCE" == false && -f "$OUT" ]] && probe_ok "$OUT"; then
        ok "Already complete → $(basename "$OUT") — skipping"
        DONE_SKIPPED=$((DONE_SKIPPED + 1))
        continue
    fi

    # A leftover but INVALID output (truncated from an interrupt) is not trusted —
    # the child will overwrite it. Note it so the log explains the re-run.
    if [[ -f "$OUT" ]]; then
        warn "Existing output is incomplete/invalid — re-running this episode"
    fi

    ep_start=$SECONDS
    # Always pass --resume: it's a no-op on a fresh episode (empty temp) and a true
    # resume on an interrupted one. -o is explicit so completion detection is stable.
    if "$UPSCALER" -i "$EP" -o "$OUT" -r "$RESOLUTION" --resume ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}; then
        # Sanity-check the child actually produced a valid file.
        if [[ -f "$OUT" ]] && probe_ok "$OUT"; then
            ep_elapsed=$((SECONDS - ep_start))
            ok "Episode $COUNT/$TOTAL done in $((ep_elapsed/3600))h$(( (ep_elapsed%3600)/60 ))m → $(basename "$OUT")"
            DONE_UPSCALED=$((DONE_UPSCALED + 1))
        else
            err "Episode $COUNT/$TOTAL: upscaler exited 0 but output is missing/invalid: $OUT"
            FAILED=$((FAILED + 1)); FAILED_LIST+=("$BASE")
            $KEEP_GOING || { err "Stopping (use --keep-going to continue past failures)."; break; }
        fi
    else
        rc=$?
        err "Episode $COUNT/$TOTAL FAILED (upscaler exit $rc): $BASE"
        FAILED=$((FAILED + 1)); FAILED_LIST+=("$BASE")
        $KEEP_GOING || { err "Stopping (use --keep-going to continue past failures). Re-run the same command to resume."; break; }
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
batch_elapsed=$((SECONDS - BATCH_START))
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
info "Batch finished in $((batch_elapsed/3600))h$(( (batch_elapsed%3600)/60 ))m"
ok  "Upscaled this run: $DONE_UPSCALED"
[[ $DONE_SKIPPED -gt 0 ]] && info "Already complete (skipped): $DONE_SKIPPED"
if [[ $FAILED -gt 0 ]]; then
    err "Failed: $FAILED — ${FAILED_LIST[*]}"
    err "Re-run the SAME command to retry failed/remaining episodes."
    exit 1
fi
ok "All ${TOTAL} episode(s) complete → $OUTPUT_DIR"
