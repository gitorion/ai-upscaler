#!/bin/bash

##############################################################################
# AI Video Upscaler — spandrel + basicsr edition
# Single-frame models: nomos8k (default), nomos8kdat, lsdir, ultrasharp, realesrgan, hat, atdjpg, nomos8kschat,
#                      spanweak, spanmedium, spanstrong, webphoto, nomos2plksr, lsdirplus
# Temporal models:     basicvsr, realbasicvsr  (multi-frame, requires basicsr)
# Optimised for live-action, compressed/noisy, artifact-heavy sources
# Requires: FFmpeg, Python 3, spandrel, CUDA GPU
##############################################################################

set -eo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$HOME/ai-upscale"
MODEL_DIR="$SCRIPT_DIR/models"
TEMP_DIR="${UPSCALE_TEMP_DIR:-$SCRIPT_DIR/temp}"   # override via env to isolate temp (e.g. A/B tests
                                                   # that must not wipe a paused run's segments)
VENV_DIR="$SCRIPT_DIR/venv"

# SeedVR2 (diffusion VSR) runs in its OWN venv with a different torch build, via the
# upstream standalone inference_cli.py. Install with prototype/seedvr2/setup.sh.
SEEDVR2_DIR="$SCRIPT_DIR/seedvr2"
SEEDVR2_VENV="$SEEDVR2_DIR/venv"
SEEDVR2_REPO="$SEEDVR2_DIR/repo"
SEEDVR2_CLI="$SEEDVR2_REPO/inference_cli.py"
SEEDVR2_MODEL_DIR="$MODEL_DIR/SEEDVR2"

# ── Defaults ──────────────────────────────────────────────────────────────────
TILE_SIZE="auto"
TILE_SIZE_EXPLICIT=false
TILE_PAD=64
MODEL_KEY="nomos8k"
QUALITY="high"
ENCODE_SPEED="slow"
PREFILTER="light"
PREFILTER_EXPLICIT=false   # tracks whether the user passed --prefilter (so per-model defaults can apply)
DEINTERLACE=false
RESUME=false
SHARPEN=false
FULL_PRECISION=false
KEEP_TEMP=false
UPSCALE_SOURCE=""
IS_TEMPORAL=false
IS_SEEDVR2=false
TEMPORAL_WINDOW="auto"
SPYNET_PATH=""

# ── SeedVR2 settings (tuned for RTX 4060 Ti 16GB) ─────────────────────────────
# Validated config from the prototype. Edit here, OR override any of these per-run via an
# environment variable, e.g.:  SEEDVR2_SEGMENT_SECONDS=60 ./upscale_video.sh -i x.mkv -r 1080p -m seedvr2
# (the ${VAR:-default} form below is what makes env overrides take effect).
SEEDVR2_MODEL_FILE="${SEEDVR2_MODEL_FILE:-seedvr2_ema_3b_fp8_e4m3fn.safetensors}"  # 3B FP8 — fits 16GB with swap
SEEDVR2_BATCH="${SEEDVR2_BATCH:-13}"            # frames/batch (4n+1); higher = better temporal consistency + more VRAM
SEEDVR2_BLOCKS_SWAP="${SEEDVR2_BLOCKS_SWAP:-16}"      # transformer blocks offloaded to CPU RAM (VRAM saver)
SEEDVR2_VAE_ENC_TILE="${SEEDVR2_VAE_ENC_TILE:-1024}"   # VAE encode tile px
SEEDVR2_VAE_DEC_TILE="${SEEDVR2_VAE_DEC_TILE:-768}"    # VAE decode tile px (1024 OOMs in decode on 16GB; 768 is the sweet spot)
SEEDVR2_TEMPORAL_OVERLAP="${SEEDVR2_TEMPORAL_OVERLAP:-3}"  # frames blended between batches/chunks — smooths seams
SEEDVR2_CHUNK="${SEEDVR2_CHUNK:-250}"           # streaming chunk size — keeps system RAM flat vs clip length (REQUIRED
                            # for long clips; whole-clip load OOM-kills). 0 = load all (short only).
# Speed optimizations that are LOSSLESS (identical output, just faster) — on by default.
# torch.compile fuses GPU kernels via Triton (already installed); does not change the math.
SEEDVR2_COMPILE="${SEEDVR2_COMPILE:-true}"        # compile the DiT (stable shapes — clean ~20-40% win)
# VAE compile is OFF by default: the VAE's tiled conv layers have many shapes, so torch 2.6's
# dynamo thrashes (hits cache_size_limit, recompiles, falls back to eager) — wasted warmup re-paid
# per segment, no speedup. Enable only if you've raised the dynamo cache limit and measured a win.
SEEDVR2_COMPILE_VAE="${SEEDVR2_COMPILE_VAE:-false}"
# Attention backend. 'auto' = use flash_attn_2 if flash-attn is installed in the SeedVR2 venv
# (lossless: exact attention, just fused, and faster), else fall back to 'sdpa' (also lossless,
# needs nothing). So to get the speedup you ONLY `pip install flash-attn` — no flag to flip.
# Pin to 'sdpa' or 'flash_attn_2' to override. Do NOT use sageattn_* — it QUANTIZES (lossy).
SEEDVR2_ATTENTION="${SEEDVR2_ATTENTION:-auto}"
# Auto-segmentation for long files. A 45-min clip is ~4-5 DAYS of compute; running it as one
# process means a single crash/reboot loses everything (the CLI has no mid-run resume). So files
# longer than this are split into segments, each upscaled independently and atomically, then
# losslessly concatenated. --resume skips already-finished segments. Larger = fewer concat seams
# but coarser checkpoints (lose more on a crash); smaller = finer checkpoints but more seams.
SEEDVR2_SEGMENT_SECONDS="${SEEDVR2_SEGMENT_SECONDS:-300}"   # 5 min. Set 0 to disable segmentation (single-shot regardless of length).

# ── Single-frame model registry (spandrel) — all 4x ──────────────────────────
declare -A MODEL_FILES=(
    [nomos8k]="4xNomos8kSC.pth"
    [nomos8kdat]="4xNomos8kDAT.pth"
    [lsdir]="4xLSDIR.pth"
    [ultrasharp]="4x-UltraSharp.pth"
    [realesrgan]="RealESRGAN_x4plus.pth"
    [hat]="HAT-L_SRx4_ImageNet-pretrain.pth"
    [atdjpg]="4xNomos8k_atd_jpg.pth"
    [nomos8kschat]="4xNomos8kSCHAT-L.pth"
    [spanweak]="4xNomos8k_span_otf_weak.pth"
    [spanmedium]="4xNomos8k_span_otf_medium.pth"
    [spanstrong]="4xNomos8k_span_otf_strong.pth"
    [webphoto]="4xNomosWebPhoto_RealPLKSR.pth"
    [nomos2plksr]="4xNomos2_realplksr_dysample.pth"
    [lsdirplus]="4xLSDIRplus.pth"
)

# ── Temporal model registry (basicsr) — all 4x ───────────────────────────────
# These models process sliding windows of frames for temporal consistency.
# Requires: pip install basicsr  +  spynet_20210409-c6c1bd09.pth in models/
declare -A TEMPORAL_MODEL_FILES=(
    [basicvsr]="BasicVSR_PlusPlus_REDS4.pth"
)

##############################################################################
# Helper functions
##############################################################################

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# True if ffprobe can read the file as valid media — used to skip corrupt extracted
# audio/subtitle streams so a bad stream can't kill the final mux.
probe_ok() { ffprobe -v error -i "$1" -show_entries format=duration -of csv=p=0 >/dev/null 2>&1; }

usage() {
    cat << EOF
${GREEN}AI Video Upscaler${NC} — spandrel + basicsr edition, live-action optimised

Usage: $0 -i INPUT [-r RESOLUTION] [OPTIONS]

  Run with just -i for interactive mode, or pass all options via CLI.

Required:
  -i, --input FILE          Input video file
  -r, --resolution RES      Target resolution: 720p, 1080p, 1440p, 2160p
                             (omit for interactive mode)

Model selection (-m / --model):
  ── Single-frame models (spandrel) ──────────────────────────────────────────
  SPAN (~0.5-1 s/frame — fastest):
  spanmedium    SPAN on Nomos8k + real-world degradation (medium)             ← best fast option
  spanweak      SPAN on Nomos8k — lighter degradation (better/cleaner sources)
  spanstrong    SPAN on Nomos8k — heavy degradation (badly compressed sources)

  RealPLKSR (~1-2 s/frame — fast):
  webphoto      RealPLKSR — web/streaming sources (lens blur + JPEG/WebP + noise)
  nomos2plksr   RealPLKSR — cleaner compressed sources (JPEG only, less aggressive)

  RRDB (~4 s/frame — standard):
  nomos8k       Best all-round for compressed live-action                     ← default
  lsdirplus     LSDIR dataset + real degradation — sharp detail on degraded sources
  lsdir         Sharp detail (clean sources)
  ultrasharp    Maximum sharpness (better on cleaner sources)
  realesrgan    Real-ESRGAN x4plus (legacy fallback)

  Transformer (20-60 s/frame — slow, short clips only):
  atdjpg        ATD — best for heavily JPEG-compressed/degraded sources
  nomos8kschat  HAT-L fine-tuned on Nomos8k — HAT quality on real-world sources
  hat           HAT-L — highest fidelity, clean sources only
  nomos8kdat    DAT — highest single-frame quality

  ── Temporal models (basicsr) — multi-frame, best temporal consistency ─────
  basicvsr    BasicVSR++ — strong on real-world degraded video  (requires basicsr)

  Temporal models process a sliding window of frames simultaneously using optical
  flow, producing sharper results with far less frame-to-frame flickering.
  Requires: pip install basicsr  +  spynet_20210409-c6c1bd09.pth in models/
  See --temporal-window to adjust window size.

  ── Diffusion VSR — highest quality on low-res/degraded sources ────────────
  seedvr2     SeedVR2 — one-step diffusion. Reconstructs (not just sharpens) detail;
              the biggest quality jump for genuinely low-res/compressed live-action.
              SLOW (~4-7 s/frame on 16GB). Long files are auto-segmented and fully
              resumable (--resume), so full episodes are practical if you have the time.
              Generative: can fabricate fine detail (watch fidelity).
              Separate install: prototype/seedvr2/setup.sh (own venv, weights auto-DL).

Pre-processing (applied before AI upscaling to clean degraded sources):
  --prefilter LEVEL         none, light (default), medium, heavy
                              none       No filtering — use for clean/high-quality sources
                                         (also the default for -m seedvr2: diffusion handles
                                          degradation itself; pre-denoise removes detail cues)
                              light      Mild temporal denoise (default — safe for most content)
                              medium     Denoise + deblock (for visibly compressed/blocky sources)
                              heavy      Strong denoise + deblock + deringing
  --deinterlace             Deinterlace source (for interlaced TV captures etc.)

Output:
  -o, --output FILE         Output file (default: INPUT_upscaled_RES.mkv)
  -q, --quality QUALITY     high (crf 16), medium (crf 20), low (crf 24) [default: high]
  --encode-speed SPEED      Encode speed/quality tradeoff [default: slow]
                              slow       Best quality, slowest encode (default)
                              medium     Balanced — ~2x faster than slow
                              fast       Quick encode — ~4x faster, slightly lower quality
  --sharpen                 Apply mild unsharp mask to final output

Performance / quality (single-frame models only):
  -t, --tile SIZE           Tile size for GPU processing [default: auto]
                            Auto mode probes GPU VRAM after model load and selects the
                            largest tile size that fits. Override with an explicit value:
                              0      Full-frame — no tiling (fastest, needs most VRAM)
                              768    Large tiles
                              512    Medium tiles
                              256    Small tiles (safe for transformer models on 16GB)
                              128    Minimum — for very low VRAM
  --tile-pad SIZE           Tile overlap padding in pixels (default: 64)
                            Increase to reduce seam artifacts; decrease to save VRAM
  --full-precision          Use float32 instead of float16 (marginal quality gain, uses more VRAM)

Temporal model options:
  --temporal-window N       Sliding window size in frames [default: auto]
                            Auto mode probes GPU VRAM and selects the largest window
                            that fits (max 15). Override with an explicit value.

Workflow:
  --resume                  Resume interrupted run — skip already-completed frames
                            (for seedvr2: skips already-finished segments of a long file)
  --keep-temp               Keep temporary files after completion
  -h, --help                Show this help

Examples:
  # Standard — compressed broadcast or web download
  $0 -i film.mkv -r 1080p

  # Temporal — best for flickery/heavily compressed sources
  $0 -i film.mkv -r 1080p -m basicvsr

  # Temporal with smaller window (reduces VRAM)
  $0 -i film.mkv -r 1080p -m basicvsr --temporal-window 7

  # Heavily compressed source (JPEG artifacts, DVD rips, old web downloads)
  $0 -i dvd_rip.mkv -r 1080p -m atdjpg --prefilter medium

  # Heavily degraded source
  $0 -i old_capture.mkv -r 1080p --prefilter heavy --deinterlace

  # Clean source, highest fidelity single-frame model
  $0 -i bluray_rip.mkv -r 2160p -m hat --prefilter none

  # Diffusion VSR — biggest quality jump on low-res clips (slow; short clips)
  $0 -i clip.mkv -r 1080p -m seedvr2

  # Resume an interrupted run
  $0 -i film.mkv -r 1080p --resume

  # 4K with sharpening, custom tile size
  $0 -i film.mkv -r 2160p -t 256 --sharpen

Model downloads (place .pth files in $MODEL_DIR):
  See README for full wget commands. Quick reference:
  nomos8k      github.com/Phhofm/models                  → 4xNomos8kSC.pth
  lsdirplus    github.com/Phhofm/models                  → 4xLSDIRplus.pth
  spanmedium   Google Drive (helaman)                     → 4xNomos8k_span_otf_medium.pth
  spanweak     Google Drive (helaman)                     → 4xNomos8k_span_otf_weak.pth
  spanstrong   Google Drive (helaman)                     → 4xNomos8k_span_otf_strong.pth
  webphoto     github.com/Phhofm/models                  → 4xNomosWebPhoto_RealPLKSR.pth
  nomos2plksr  github.com/Phhofm/models                  → 4xNomos2_realplksr_dysample.pth
  nomos8kdat   openmodeldb.info                           → 4xNomos8kDAT.pth
  lsdir        openmodeldb.info/models/4x-LSDIR           → 4xLSDIR.pth
  ultrasharp   huggingface.co/Kim2091/UltraSharp          → 4x-UltraSharp.pth
  realesrgan   github.com/xinntao/Real-ESRGAN             → RealESRGAN_x4plus.pth
  atdjpg       github.com/Phhofm/models                  → 4xNomos8k_atd_jpg.pth
  nomos8kschat Google Drive (Phhofm)                     → 4xNomos8kSCHAT-L.pth
  hat          huggingface.co/anchuang/HAT-L_SRx4_ImageNet-pretrain → HAT-L_SRx4_ImageNet-pretrain.pth
               (requires: pip install spandrel-extra-arches)
  basicvsr     openmmlab CDN                              → BasicVSR_PlusPlus_REDS4.pth
               wget -O ~/ai-upscale/models/BasicVSR_PlusPlus_REDS4.pth \
               https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth
               (requires: pip install basicsr)
  spynet       Required by both temporal models           → spynet_20210409-c6c1bd09.pth
               (downloaded automatically by basicsr on first use)
  seedvr2      Separate install — own venv + standalone CLI (NOT a .pth in models/)
               Setup:   <repo>/prototype/seedvr2/setup.sh
               Weights: auto-downloaded to models/SEEDVR2/ on first run (~3.6GB)
EOF
    exit 0
}

check_dependencies() {
    print_info "Checking dependencies..."

    for cmd in ffmpeg ffprobe bc; do
        if ! command -v "$cmd" &>/dev/null; then
            print_error "$cmd not found — install with: sudo apt install ffmpeg bc"
            exit 1
        fi
    done

    if ! command -v nvidia-smi &>/dev/null; then
        print_warning "nvidia-smi not found — GPU unavailable; CPU upscaling will be very slow"
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        print_error "Virtual environment not found at $VENV_DIR"
        print_error "Run the setup script first"
        exit 1
    fi

    local python="$VENV_DIR/bin/python3"

    if ! "$python" -c "import spandrel" &>/dev/null; then
        print_error "spandrel not installed in venv"
        print_error "Fix: source $VENV_DIR/bin/activate && pip install spandrel spandrel-extra-arches"
        exit 1
    fi

    for pkg in cv2 torch tqdm numpy; do
        if ! "$python" -c "import $pkg" &>/dev/null; then
            print_error "Python package '$pkg' not found in venv"
            exit 1
        fi
    done

    print_success "Dependencies OK"
}

get_video_info() {
    local f="$1"
    print_info "Analysing input..."

    INPUT_WIDTH=$(ffprobe  -v error -select_streams v:0 -show_entries stream=width            -of default=noprint_wrappers=1:nokey=1 "$f" | head -1)
    INPUT_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height           -of default=noprint_wrappers=1:nokey=1 "$f" | head -1)
    INPUT_FPS=$(ffprobe    -v error -select_streams v:0 -show_entries stream=r_frame_rate     -of default=noprint_wrappers=1:nokey=1 "$f" | head -1 | bc -l | xargs printf "%.3f")
    DURATION=$(ffprobe     -v error                     -show_entries format=duration         -of default=noprint_wrappers=1:nokey=1 "$f" | head -1 | xargs printf "%.2f")
    # Cascading frame count: nb_frames (container metadata, instant) → count_packets (fast) → count_frames (slow, full decode)
    TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 "$f" | head -1)
    if [[ -z "$TOTAL_FRAMES" || "$TOTAL_FRAMES" == "N/A" || "$TOTAL_FRAMES" -le 0 ]] 2>/dev/null; then
        TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of default=noprint_wrappers=1:nokey=1 "$f" | head -1)
    fi
    if [[ -z "$TOTAL_FRAMES" || "$TOTAL_FRAMES" == "N/A" || "$TOTAL_FRAMES" -le 0 ]] 2>/dev/null; then
        TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$f" | head -1)
    fi
    INPUT_CODEC=$(ffprobe  -v error -select_streams v:0 -show_entries stream=codec_name       -of default=noprint_wrappers=1:nokey=1 "$f" | head -1)

    # SAR / DAR — needed for correct output width on anamorphic (non-square pixel) sources
    INPUT_SAR=$(ffprobe -v error -select_streams v:0 -show_entries stream=sample_aspect_ratio  -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null | head -1 || echo "")
    INPUT_DAR=$(ffprobe -v error -select_streams v:0 -show_entries stream=display_aspect_ratio -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null | head -1 || echo "")

    # Field order — detect interlaced vs progressive
    INPUT_FIELD_ORDER=$(ffprobe -v error -select_streams v:0 -show_entries stream=field_order -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null | head -1 || echo "")
    case "$INPUT_FIELD_ORDER" in
        tt|bb|tb|bt) IS_INTERLACED=true  ;;
        *)           IS_INTERLACED=false ;;
    esac

    local audio_stream
    audio_stream=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null | head -1 || true)
    HAS_AUDIO=$([[ -n "$audio_stream" ]] && echo true || echo false)

    local sub_stream
    sub_stream=$(ffprobe -v error -select_streams s:0 -show_entries stream=codec_type -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null | head -1 || true)
    HAS_SUBS=$([[ -n "$sub_stream" ]] && echo true || echo false)

    local scan_label="progressive"
    [[ "$IS_INTERLACED" == true ]] && scan_label="interlaced"
    print_info "Input: ${INPUT_WIDTH}x${INPUT_HEIGHT}  SAR:${INPUT_SAR:-1:1}  DAR:${INPUT_DAR:-N/A}  ${INPUT_FPS}fps  ${DURATION}s  ${TOTAL_FRAMES} frames  codec:${INPUT_CODEC}  scan:${scan_label}  audio:${HAS_AUDIO}  subs:${HAS_SUBS}"
}

calculate_scale() {
    local target_res="$1"

    case "$target_res" in
        720p)  TARGET_HEIGHT=720  ;;
        1080p) TARGET_HEIGHT=1080 ;;
        1440p) TARGET_HEIGHT=1440 ;;
        2160p) TARGET_HEIGHT=2160 ;;
        *)
            print_error "Invalid resolution: $target_res (use 720p, 1080p, 1440p, 2160p)"
            exit 1
            ;;
    esac

    SCALE_FACTOR=$(echo "scale=6; $TARGET_HEIGHT / $INPUT_HEIGHT" | bc)

    if (( $(echo "$SCALE_FACTOR <= 1.0" | bc -l) )); then
        print_warning "Input ($INPUT_HEIGHT) >= target ($TARGET_HEIGHT) — will downscale with FFmpeg only"
        USE_AI=false
    else
        USE_AI=true
    fi

    OUTPUT_HEIGHT=$(( (TARGET_HEIGHT / 2) * 2 ))

    # Use DAR (display aspect ratio) for output width when available.
    # This preserves the correct shape for anamorphic sources — e.g. SD DVD content
    # stored as 720x480 (3:2 pixels) but intended to display as 4:3 (SAR 8:9).
    # For square-pixel sources (SAR 1:1) DAR equals the pixel ratio, so the result
    # is identical to the old calculation.
    local dar_note=""
    local dar_w dar_h
    if [[ -n "$INPUT_DAR" && "$INPUT_DAR" != "N/A" && "$INPUT_DAR" != "0:1" ]]; then
        dar_w=$(echo "$INPUT_DAR" | cut -d: -f1)
        dar_h=$(echo "$INPUT_DAR" | cut -d: -f2)
        if [[ -n "$dar_w" && -n "$dar_h" && "$dar_h" -gt 0 ]]; then
            OUTPUT_WIDTH=$(echo "$OUTPUT_HEIGHT * $dar_w / $dar_h" | bc | cut -d. -f1)
            OUTPUT_WIDTH=$(( (OUTPUT_WIDTH / 2) * 2 ))
            dar_note="  (DAR ${INPUT_DAR})"
        fi
    fi

    if [[ -z "$dar_note" ]]; then
        OUTPUT_WIDTH=$(echo "$INPUT_WIDTH * $TARGET_HEIGHT / $INPUT_HEIGHT" | bc | cut -d. -f1)
        OUTPUT_WIDTH=$(( (OUTPUT_WIDTH / 2) * 2 ))
    fi

    print_info "Scale factor: ${SCALE_FACTOR}x  →  Output: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}${dar_note}"
}

select_model() {
    # SeedVR2 (diffusion VSR) — separate venv + standalone CLI, not a .pth in MODEL_DIR
    if [[ "$MODEL_KEY" == "seedvr2" ]]; then
        IS_SEEDVR2=true
        IS_TEMPORAL=false
        if [[ ! -f "$SEEDVR2_CLI" ]]; then
            print_error "SeedVR2 not installed: $SEEDVR2_CLI not found"
            print_error "Install it first:  cd <repo>/prototype/seedvr2 && ./setup.sh"
            exit 1
        fi
        if [[ ! -x "$SEEDVR2_VENV/bin/python3" ]]; then
            print_error "SeedVR2 venv missing: $SEEDVR2_VENV"
            print_error "Run prototype/seedvr2/setup.sh to create it"
            exit 1
        fi
        print_info "Model: seedvr2  (${SEEDVR2_MODEL_FILE})  [diffusion VSR — separate venv]"
        return
    fi

    # Check single-frame registry (spandrel)
    if [[ -n "${MODEL_FILES[$MODEL_KEY]+_}" ]]; then
        IS_TEMPORAL=false
        MODEL_PATH="$MODEL_DIR/${MODEL_FILES[$MODEL_KEY]}"
        if [[ ! -f "$MODEL_PATH" ]]; then
            print_error "Model not found: $MODEL_PATH"
            print_error "Download it and place in: $MODEL_DIR  (see --help for URLs)"
            exit 1
        fi
        print_info "Model: $MODEL_KEY  (${MODEL_FILES[$MODEL_KEY]})  [single-frame]"
        return
    fi

    # Check temporal registry (basicsr)
    if [[ -n "${TEMPORAL_MODEL_FILES[$MODEL_KEY]+_}" ]]; then
        IS_TEMPORAL=true
        MODEL_PATH="$MODEL_DIR/${TEMPORAL_MODEL_FILES[$MODEL_KEY]}"
        SPYNET_PATH="$MODEL_DIR/spynet_20210409-c6c1bd09.pth"
        if [[ ! -f "$MODEL_PATH" ]]; then
            print_error "Temporal model not found: $MODEL_PATH"
            print_error "Download it and place in: $MODEL_DIR  (see --help for URLs)"
            exit 1
        fi
        if [[ ! -f "$SPYNET_PATH" ]]; then
            print_warning "SPyNet weights not found at: $SPYNET_PATH"
            print_warning "basicsr will attempt to download them automatically on first run."
            print_warning "If this fails: download spynet_20210409-c6c1bd09.pth manually to $MODEL_DIR"
            SPYNET_PATH=""  # let basicsr handle auto-download
        fi
        print_info "Model: $MODEL_KEY  (${TEMPORAL_MODEL_FILES[$MODEL_KEY]})  [temporal/multi-frame]"
        return
    fi

    print_error "Unknown model: $MODEL_KEY"
    print_error "Single-frame (spandrel): ${!MODEL_FILES[*]}"
    print_error "Temporal (basicsr):      ${!TEMPORAL_MODEL_FILES[*]}"
    print_error "Diffusion VSR:           seedvr2"
    exit 1
}

run_prefilter() {
    local source_file="$1"
    local cleaned="$TEMP_DIR/cleaned_source.mkv"
    UPSCALE_SOURCE="$source_file"

    local vf_parts=()

    if [[ "$DEINTERLACE" == true ]]; then
        if [[ "$IS_INTERLACED" == true ]]; then
            # mode=1: output a frame for each field (doubles framerate to preserve temporal detail)
            vf_parts+=("yadif=mode=1:parity=-1:deint=1")
            print_info "Deinterlace: mode=1 (interlaced source — doubling framerate)"
        else
            # mode=0: frame-rate preserving — safe no-op for progressive content
            vf_parts+=("yadif=mode=0:parity=-1:deint=1")
            print_info "Deinterlace: mode=0 (progressive source — framerate preserved)"
        fi
    fi

    case "$PREFILTER" in
        none)   ;;
        light)  vf_parts+=("hqdn3d=1.5:1:1.5:1") ;;
        medium) vf_parts+=("hqdn3d=3:2:3:2,pp=hb/vb") ;;
        heavy)  vf_parts+=("hqdn3d=6:4:6:4,pp=hb/vb/dr") ;;
        *)
            print_error "Unknown prefilter: $PREFILTER (use none, light, medium, heavy)"
            exit 1
            ;;
    esac

    if [[ ${#vf_parts[@]} -eq 0 ]]; then
        return  # no filtering — upscale directly from source
    fi

    # On resume, reuse the existing cleaned source if present
    if [[ "$RESUME" == true && -f "$cleaned" ]]; then
        print_info "Resume: reusing existing cleaned_source.mkv"
        UPSCALE_SOURCE="$cleaned"
        return
    fi

    local vf_string
    vf_string=$(IFS=,; echo "${vf_parts[*]}")
    print_info "Pre-filtering: $vf_string"

    # ffv1 (lossless) avoids re-introducing compression artifacts before the AI sees the frame
    ffmpeg -i "$source_file" \
        -vf "$vf_string" \
        -c:v ffv1 -level 3 \
        -an \
        "$cleaned" -y -loglevel error

    UPSCALE_SOURCE="$cleaned"
    print_success "Pre-filter complete ($(du -h "$cleaned" | cut -f1))"
}

extract_audio() {
    local input_file="$1"
    local audio_file="$TEMP_DIR/audio.mka"

    if [[ "$HAS_AUDIO" == false ]]; then
        return
    fi

    # On resume, reuse existing audio extraction
    if [[ "$RESUME" == true && -f "$audio_file" && -s "$audio_file" ]]; then
        print_info "Resume: reusing existing audio"
        return
    fi

    print_info "Extracting audio..."
    # Extract all audio streams from original input (cleaned_source.mkv has -an / no audio)
    ffmpeg -i "$input_file" -vn -sn -map 0:a -c:a copy "$audio_file" -y -loglevel error || true

    if [[ -f "$audio_file" && -s "$audio_file" ]] && probe_ok "$audio_file"; then
        print_success "Audio extracted"
    else
        HAS_AUDIO=false
        print_warning "Audio extraction failed or unreadable — encoding without audio"
    fi
}

extract_subs() {
    local input_file="$1"
    local subs_file="$TEMP_DIR/subs.mkv"

    if [[ "$HAS_SUBS" == false ]]; then
        return
    fi

    # On resume, reuse existing subtitle extraction
    if [[ "$RESUME" == true && -f "$subs_file" && -s "$subs_file" ]]; then
        print_info "Resume: reusing existing subtitles"
        return
    fi

    print_info "Extracting subtitles..."
    # Extract all subtitle streams from original input
    ffmpeg -i "$input_file" -vn -an -map 0:s -c:s copy "$subs_file" -y -loglevel error || true

    if [[ -f "$subs_file" && -s "$subs_file" ]] && probe_ok "$subs_file"; then
        print_success "Subtitles extracted"
    else
        HAS_SUBS=false
        print_warning "Subtitle extraction failed or unreadable — encoding without subtitles"
    fi
}

write_python_script() {
    cat > "$TEMP_DIR/upscale.py" << 'PYTHON_EOF'
import sys
import os
import math
import threading
import queue
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# HAT and other transformer models may require spandrel-extra-arches
try:
    import spandrel_extra_arches
    spandrel_extra_arches.install()
except ImportError:
    pass

from spandrel import ImageModelDescriptor, ModelLoader

# Allow cuDNN to benchmark and select the fastest convolution algorithm for
# the actual tile dimensions in use. Pays off quickly on video — all frames
# share the same tile sizes, so the benchmark runs only once per shape.
torch.backends.cudnn.benchmark = True

# Transformer models (HAT, SwinIR) require input H/W to be a multiple of their
# window size. 64 is a safe LCM covering all supported architectures.
WINDOW_MULTIPLE = 64

# Pipeline queue depths — tune if you hit OOM or want more IO/GPU overlap.
# Each slot in read_q holds one raw decoded frame; write_q holds one upscaled frame.
READER_QUEUE_DEPTH = 8
WRITER_QUEUE_DEPTH = 16


def pad_to_multiple(tensor, multiple):
    """Pad H and W dims of an NCHW tensor to the nearest multiple. Returns (padded, pad_w, pad_h)."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, 0, 0
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect'), pad_w, pad_h


def frame_to_tensor(frame, device, use_half):
    """Convert cv2 BGR uint8 HWC frame to NCHW float [0,1] tensor on device."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    if use_half:
        t = t.half()
    # non_blocking=True enables async host→device DMA when the source tensor
    # is in pinned memory — harmless no-op otherwise.
    return t.to(device, non_blocking=True)


def tensor_to_frame(t, frame_num=None):
    """Convert NCHW float [0,1] tensor to cv2 BGR uint8 HWC frame."""
    raw = t.squeeze(0).float()
    if not torch.isfinite(raw).all():
        bad = (~torch.isfinite(raw)).sum().item()
        total = raw.numel()
        label = f"frame {frame_num}" if frame_num is not None else "frame"
        if bad == total:
            print(
                f"\n[WARN] {label}: entire output is NaN/Inf — model is overflowing float16. "
                "Re-run with --full-precision to fix this (use -t 512 to keep within VRAM).",
                flush=True,
            )
        else:
            print(f"\n[WARN] {label}: {bad}/{total} NaN/Inf values replaced with 0.", flush=True)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
    out = raw.cpu().clamp(0, 1)
    return cv2.cvtColor(
        (out.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )


def tile_process(model, img_tensor, model_scale, tile_size, tile_pad):
    """
    Tile-based inference with per-tile WINDOW_MULTIPLE alignment padding.

    Each tile borrows tile_pad pixels of context from its neighbours for better
    edge quality, then is further padded to WINDOW_MULTIPLE for transformer
    compatibility (harmless no-op for RRDBNet-based models).

    The accumulation buffer stays on the GPU — a single CPU transfer happens
    at the end in tensor_to_frame(), instead of one per tile.
    """
    batch, channel, height, width = img_tensor.shape
    device = img_tensor.device

    # Keep output on GPU to avoid per-tile round-trips
    output = torch.zeros(
        batch, channel, height * model_scale, width * model_scale,
        dtype=torch.float32, device=device,
    )

    tiles_x = math.ceil(width  / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            # Core tile bounds (no padding)
            x0 = tile_x * tile_size;  x1 = min(x0 + tile_size, width)
            y0 = tile_y * tile_size;  y1 = min(y0 + tile_size, height)

            # Expanded bounds with tile_pad context for better edge quality
            x0p = max(x0 - tile_pad, 0);  x1p = min(x1 + tile_pad, width)
            y0p = max(y0 - tile_pad, 0);  y1p = min(y1 + tile_pad, height)

            tile_in = img_tensor[:, :, y0p:y1p, x0p:x1p]
            tile_in, pw, ph = pad_to_multiple(tile_in, WINDOW_MULTIPLE)

            with torch.no_grad():
                tile_out = model(tile_in).float()

            # Strip WINDOW_MULTIPLE padding from output
            tile_out = tile_out[
                :, :,
                : tile_out.shape[2] - ph * model_scale,
                : tile_out.shape[3] - pw * model_scale,
            ]

            # Place only the core (non-tile_pad) region into the accumulation buffer
            off_x  = (x0  - x0p) * model_scale
            off_y  = (y0  - y0p) * model_scale
            w_tile = (x1  - x0)  * model_scale
            h_tile = (y1  - y0)  * model_scale

            output[
                :, :,
                y0 * model_scale : y1 * model_scale,
                x0 * model_scale : x1 * model_scale,
            ] = tile_out[:, :, off_y : off_y + h_tile, off_x : off_x + w_tile]

    return output


def probe_tile_size(model, model_scale, src_h, src_w, tile_pad, device, use_half):
    """Try descending tile sizes on a dummy frame to find the largest that fits in VRAM."""
    candidates = [0, 768, 512, 384, 256, 192, 128]

    dtype = torch.float16 if use_half else torch.float32
    dummy = torch.rand(1, 3, src_h, src_w, dtype=dtype, device=device)
    print(f"Probing optimal tile size for {src_w}x{src_h} input...", flush=True)

    with torch.no_grad():
        for size in candidates:
            torch.cuda.empty_cache()
            try:
                if size == 0:
                    img_pad, pw, ph = pad_to_multiple(dummy, WINDOW_MULTIPLE)
                    out = model(img_pad).float()
                    del out, img_pad
                else:
                    out = tile_process(model, dummy, model_scale, size, tile_pad)
                    del out
                del dummy
                torch.cuda.empty_cache()
                label = "full-frame" if size == 0 else str(size)
                print(f"Auto tile: {label} — OK", flush=True)
                return size
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    label = "full-frame" if size == 0 else str(size)
                    print(f"Auto tile: {label} — OOM, trying smaller", flush=True)
                    continue
                raise

    del dummy
    torch.cuda.empty_cache()
    print("Auto tile: all sizes OOM — using 128 as minimum", flush=True)
    return 128


def upscale_video(input_video, frames_dir, model_path,
                  tile_size, tile_pad, output_w, output_h,
                  use_full_precision, resume):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_half = (not use_full_precision) and (device == 'cuda')
    print(f"Device: {device}  |  Precision: {'float16' if use_half else 'float32'}")

    print(f"Loading model: {os.path.basename(model_path)}")
    descriptor = ModelLoader().load_from_file(model_path)
    if not isinstance(descriptor, ImageModelDescriptor):
        raise RuntimeError(f"Not an image super-resolution model: {model_path}")

    model_scale = descriptor.scale
    model = descriptor.model.eval().to(device)
    if use_half:
        model = model.half()

    print(f"Model scale: {model_scale}x  |  Target output: {output_w}x{output_h}")

    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Auto-probe optimal tile size if requested
    if tile_size < 0:
        tile_size = probe_tile_size(model, model_scale, src_h, src_w, tile_pad, device, use_half)
    tile_label = "full-frame" if tile_size == 0 else str(tile_size)
    print(f"Tile size: {tile_label}", flush=True)

    # Use ffprobe for reliable fps/frame-count (OpenCV is unreliable on MKV/FFV1)
    import subprocess, json as _json

    # Get fps and dimensions from metadata (instant)
    _probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate,nb_frames',
         '-of', 'json', input_video],
        capture_output=True, text=True)
    _info = _json.loads(_probe.stdout)['streams'][0]
    _num, _den = map(int, _info['r_frame_rate'].split('/'))
    fps = _num / _den

    # Cascading frame count: nb_frames (container, instant) → count_packets (fast) → count_frames (slow)
    total_frames = int(_info.get('nb_frames', 0) or 0)
    if total_frames <= 0:
        _probe2 = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
             '-show_entries', 'stream=nb_read_packets', '-of', 'json', input_video],
            capture_output=True, text=True)
        _info2 = _json.loads(_probe2.stdout)['streams'][0]
        total_frames = int(_info2.get('nb_read_packets', 0) or 0)
    if total_frames <= 0:
        _probe3 = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
             '-show_entries', 'stream=nb_read_frames', '-of', 'json', input_video],
            capture_output=True, text=True)
        _info3 = _json.loads(_probe3.stdout)['streams'][0]
        total_frames = int(_info3.get('nb_read_frames', 0) or 0)
    total_frames = total_frames or None

    with open(os.path.join(frames_dir, 'fps.txt'), 'w') as fh:
        fh.write(f"{fps:.6f}")

    # ── Async 3-stage pipeline ─────────────────────────────────────────────────
    #   Stage 1 (reader thread):  decode frames from video  → read_q
    #   Stage 2 (main / GPU):     AI upscale                read_q → write_q
    #   Stage 3 (writer thread):  compress & write PNG      write_q → disk
    #
    # This keeps the GPU busy while the previous frame is being written to disk
    # and the next frame is being decoded — especially beneficial on fast GPUs
    # or when writing to slower storage.
    # ──────────────────────────────────────────────────────────────────────────
    read_q  = queue.Queue(maxsize=READER_QUEUE_DEPTH)
    write_q = queue.Queue(maxsize=WRITER_QUEUE_DEPTH)

    skipped = [0]
    err     = threading.Event()

    def reader_fn():
        """Decode video frames and push them onto read_q. Runs in a background thread."""
        fn = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(frames_dir, f'frame_{fn:08d}.png')
                # Signal GPU stage to skip frames already on disk (resume mode)
                if resume and os.path.isfile(frame_path):
                    read_q.put((fn, None, frame_path))
                else:
                    read_q.put((fn, frame, frame_path))
                fn += 1
        except Exception as e:
            print(f"\n[ERROR] Reader thread: {e}", flush=True)
            err.set()
        finally:
            read_q.put(None)  # sentinel — signals GPU stage that decoding is done

    def writer_fn():
        """Write upscaled PNG frames from write_q to disk. Runs in a background thread."""
        try:
            while True:
                item = write_q.get()
                if item is None:
                    break
                frame_path, out_frame = item
                # PNG compression=1: fastest write, larger file — correct for temp frames
                cv2.imwrite(frame_path, out_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        except Exception as e:
            print(f"\n[ERROR] Writer thread: {e}", flush=True)
            err.set()

    reader_t = threading.Thread(target=reader_fn, daemon=True)
    writer_t = threading.Thread(target=writer_fn, daemon=True)
    reader_t.start()
    writer_t.start()

    frame_count = 0
    consecutive_fails = 0

    with tqdm(total=total_frames, desc="Upscaling frames", unit="frame") as pbar:
        while True:
            item = read_q.get()
            if item is None or err.is_set():
                break

            frame_num, frame, frame_path = item

            if frame is None:
                # Resume mode — frame already exists on disk, nothing to do
                skipped[0] += 1
                pbar.update(1)
                frame_count += 1
                continue

            try:
                h, w  = frame.shape[:2]
                img_t = frame_to_tensor(frame, device, use_half)

                if tile_size > 0:
                    out_t = tile_process(model, img_t, model_scale, tile_size, tile_pad)
                else:
                    # Full-image inference (only viable for small frames / large VRAM)
                    img_pad, pw, ph = pad_to_multiple(img_t, WINDOW_MULTIPLE)
                    with torch.no_grad():
                        out_t = model(img_pad).float()
                    if ph > 0 or pw > 0:
                        out_t = out_t[
                            :, :,
                            : out_t.shape[2] - ph * model_scale,
                            : out_t.shape[3] - pw * model_scale,
                        ]

                out_frame = tensor_to_frame(out_t, frame_num)

                # Resize from model's native output to the target display dimensions.
                # INTER_AREA for downscaling (4x native → smaller target, e.g. 1080p→2160p),
                # INTER_LANCZOS4 for upscaling (target larger than model native output).
                if out_frame.shape[0] != output_h or out_frame.shape[1] != output_w:
                    interp = cv2.INTER_AREA if (output_h < h * model_scale) else cv2.INTER_LANCZOS4
                    out_frame = cv2.resize(out_frame, (output_w, output_h), interpolation=interp)

                write_q.put((frame_path, out_frame))

            except Exception as e:
                consecutive_fails += 1
                print(f"\n[WARN] Frame {frame_num} failed ({e}) — writing resized original", flush=True)
                if consecutive_fails >= 10:
                    print(f"\n[ERROR] {consecutive_fails} consecutive frames failed — aborting.", flush=True)
                    print(f"[ERROR] Try a smaller tile size: -t 512 or -t 256", flush=True)
                    sys.exit(1)
                try:
                    fallback = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_LANCZOS4)
                    write_q.put((frame_path, fallback))
                except Exception:
                    pass
            else:
                consecutive_fails = 0

            frame_count += 1
            pbar.update(1)

    write_q.put(None)   # signal writer thread to finish
    writer_t.join()     # wait for all frames to be written to disk
    reader_t.join(timeout=2)
    cap.release()

    if skipped[0] > 0:
        print(f"Resumed: skipped {skipped[0]} already-completed frames")

    return frame_count > 0


if __name__ == '__main__':
    _, input_video, frames_dir, model_path, \
        tile_size, tile_pad, output_w, output_h, \
        use_full_precision_str, resume_str = sys.argv

    success = upscale_video(
        input_video        = input_video,
        frames_dir         = frames_dir,
        model_path         = model_path,
        tile_size          = -1 if tile_size == 'auto' else int(tile_size),
        tile_pad           = int(tile_pad),
        output_w           = int(output_w),
        output_h           = int(output_h),
        use_full_precision = (use_full_precision_str == 'true'),
        resume             = (resume_str == 'true'),
    )
    sys.exit(0 if success else 1)
PYTHON_EOF
}

write_temporal_python_script() {
    cat > "$TEMP_DIR/upscale_temporal.py" << 'TEMPORAL_EOF'
import sys
import os
import re
import importlib
# Reduce CUDA memory fragmentation. BasicVSR++ makes many large allocations during
# propagation; expandable segments lets PyTorch reuse fragmented reserved blocks.
# MUST be set before torch is imported.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import cv2
import torch
import numpy as np
from tqdm import tqdm

# ── Neuter basicsr's __init__ to avoid broken wildcard imports ──────────────────
# basicsr 1.4.2's __init__.py does `from .data import *`, `from .losses import *`,
# etc.  Several of these trigger imports of torchvision internals that were removed
# in newer torchvision (e.g. torchvision.transforms.functional_tensor).
# We only need basicsr.archs, so we replace the package __init__ entirely.
import importlib, types as _types, sys
_basicsr_spec = importlib.util.find_spec('basicsr')
if _basicsr_spec is not None:
    _basicsr_mod = _types.ModuleType('basicsr')
    _basicsr_mod.__path__ = _basicsr_spec.submodule_search_locations or [_basicsr_spec.origin.rsplit('/', 1)[0]]
    _basicsr_mod.__package__ = 'basicsr'
    _basicsr_mod.__version__ = '1.4.2'
    sys.modules['basicsr'] = _basicsr_mod

# ── Patch basicsr flow_warp for half-precision compatibility ───────────────────
# basicsr's flow_warp() explicitly casts its coordinate grid to float32 (.float()),
# which causes F.grid_sample to fail when feature maps are float16.
# spynet_arch and basicvsrpp_arch both do `from arch_util import flow_warp` at
# import time, so we must patch the name in each module's own namespace.
import basicsr.archs.arch_util as _bsr_arch_util
import basicsr.archs.spynet_arch as _bsr_spynet_arch
import basicsr.archs.basicvsrpp_arch as _bsr_basicvsrpp_arch
_orig_flow_warp = _bsr_arch_util.flow_warp
def _flow_warp_half_safe(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    orig_dtype = x.dtype
    if orig_dtype != torch.float32:
        x = x.float()
        flow = flow.float()
    out = _orig_flow_warp(x, flow, interp_mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
    return out.to(orig_dtype)
_bsr_arch_util.flow_warp = _flow_warp_half_safe
_bsr_spynet_arch.flow_warp = _flow_warp_half_safe
_bsr_basicvsrpp_arch.flow_warp = _flow_warp_half_safe

# ── Temporal model configs ─────────────────────────────────────────────────────
# arch_module / arch_class confirmed against basicsr 1.4.2 source.
# ckpt_prefix: layer name prefix added by the training framework's wrapper class
#   that must be stripped before load_state_dict. basicsr checkpoints use no
#   prefix; mmediting-trained checkpoints wrap the generator as 'generator.*'.
TEMPORAL_MODEL_CONFIGS = {
    'basicvsr': {
        'arch_module': 'basicsr.archs.basicvsrpp_arch',   # note: no underscore between vsr/pp
        'arch_class':  'BasicVSRPlusPlus',
        'params': {'mid_channels': 64, 'num_blocks': 7, 'is_low_res_input': True},
        'ckpt_prefix': 'generator.',   # mmediting wraps BasicVSRPlusPlus in a BasicVSR model
        'remap': 'basicvsrpp_mmediting',  # key names differ between mmediting and basicsr 1.4.x
    },
}

MODEL_SCALE = 4   # all supported temporal models output 4×


def _remap_basicvsrpp_mmediting(state):
    """
    Remap mmediting-era BasicVSR++ checkpoint keys to match current basicsr 1.4.x arch.

    Two structural changes between the mmediting training codebase and current basicsr:

    1. SpyNet: old arch used ConvModule wrappers, adding a '.conv.' sub-key.
       Current basicsr uses a plain nn.Sequential where Conv2d sits at even indices.
         old: spynet.basic_module.N.basic_module.M.conv.{weight,bias}
         new: spynet.basic_module.N.basic_module.{M*2}.{weight,bias}

    2. Upsampling: old arch used PixelShufflePack (.upsample_conv).
       Current basicsr uses plain ConvTranspose2d named upconv1/upconv2.
         old: upsample{N}.upsample_conv.{weight,bias}
         new: upconv{N}.{weight,bias}
    """
    spynet_re = re.compile(
        r'^(spynet\.basic_module\.\d+\.basic_module\.)(\d+)\.conv\.(weight|bias)$'
    )
    upsample_re = re.compile(r'^upsample(\d+)\.upsample_conv\.(weight|bias)$')
    out = {}
    for k, v in state.items():
        m = spynet_re.match(k)
        if m:
            out[f"{m.group(1)}{int(m.group(2)) * 2}.{m.group(3)}"] = v
            continue
        m = upsample_re.match(k)
        if m:
            out[f"upconv{m.group(1)}.{m.group(2)}"] = v
            continue
        out[k] = v
    return out


def load_temporal_model(model_key, model_path, spynet_path, device, use_half):
    cfg = TEMPORAL_MODEL_CONFIGS[model_key]
    module = importlib.import_module(cfg['arch_module'])
    arch_cls = getattr(module, cfg['arch_class'])

    params = dict(cfg['params'])
    # Pass spynet_path=None — the mmediting REDS4 checkpoint embeds SPyNet weights
    # under generator.spynet.*, loaded via load_state_dict below.
    params['spynet_path'] = None

    model = arch_cls(**params)

    ckpt = torch.load(model_path, map_location='cpu')

    # Extract weights from the top-level checkpoint key (basicsr uses 'params'/'params_ema';
    # mmediting-trained checkpoints use 'state_dict')
    state = ckpt
    for key in ('params_ema', 'params', 'state_dict'):
        if key in ckpt:
            state = ckpt[key]
            break

    # Strip any wrapper prefix added by the training framework.
    # mmediting wraps BasicVSRPlusPlus inside a BasicVSR model as self.generator,
    # so all keys are prefixed 'generator.' in the checkpoint.
    prefix = cfg.get('ckpt_prefix', '')
    if prefix:
        state = {k[len(prefix):]: v
                 for k, v in state.items()
                 if k.startswith(prefix)}

    # Remap mmediting-era key names to match current basicsr arch definitions.
    if cfg.get('remap') == 'basicvsrpp_mmediting':
        state = _remap_basicvsrpp_mmediting(state)

    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    if use_half:
        model = model.half()

    return model


def frame_to_tensor(frame, device, use_half):
    """cv2 BGR uint8 HWC → CHW float [0,1] on device."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1)
    if use_half:
        t = t.half()
    return t.to(device, non_blocking=True)


def tensor_to_frame(t):
    """CHW float tensor → cv2 BGR uint8 HWC frame."""
    raw = t.float()
    if not torch.isfinite(raw).all():
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
    out = raw.cpu().clamp(0, 1)
    return cv2.cvtColor(
        (out.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )


def probe_window_size(model, window_size, src_h, src_w, device, use_half):
    """Try descending window sizes on dummy frames to find the largest that fits in VRAM."""
    candidates = [s for s in [window_size, 13, 11, 9, 7, 5, 3] if s <= window_size]
    # Deduplicate while preserving order
    seen = set()
    candidates = [s for s in candidates if not (s in seen or seen.add(s))]

    dtype = torch.float16 if use_half else torch.float32
    print(f"Probing optimal window size for {src_w}x{src_h} input...", flush=True)

    with torch.no_grad():
        for size in candidates:
            torch.cuda.empty_cache()
            try:
                dummy = torch.rand(1, size, 3, src_h, src_w, dtype=dtype, device=device)
                out = model(dummy)
                del out, dummy
                torch.cuda.empty_cache()
                print(f"Auto window: {size} frames — OK", flush=True)
                return size
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    try:
                        del dummy
                    except NameError:
                        pass
                    torch.cuda.empty_cache()
                    print(f"Auto window: {size} frames — OOM, trying smaller", flush=True)
                    continue
                raise

    torch.cuda.empty_cache()
    print(f"Auto window: all sizes OOM — using {candidates[-1]} as minimum", flush=True)
    return candidates[-1]


def upscale_temporal(input_video, frames_dir, model_key, model_path, spynet_path,
                     window_size, output_w, output_h, use_full_precision, resume):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_half = (not use_full_precision) and (device == 'cuda')
    print(f"Device: {device}  |  Precision: {'float16' if use_half else 'float32'}")
    print(f"Loading temporal model: {os.path.basename(model_path)}")

    model = load_temporal_model(model_key, model_path, spynet_path, device, use_half)

    os.makedirs(frames_dir, exist_ok=True)

    # Use ffprobe for reliable fps, frame-count, and dimensions
    import subprocess, json as _json

    # Get fps, dimensions, and container frame count (instant)
    _probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate,nb_frames,width,height',
         '-of', 'json', input_video],
        capture_output=True, text=True)
    _info = _json.loads(_probe.stdout)['streams'][0]
    _num, _den = map(int, _info['r_frame_rate'].split('/'))
    fps = _num / _den
    src_w = int(_info['width'])
    src_h = int(_info['height'])

    # Cascading frame count: nb_frames (container, instant) → count_packets (fast) → count_frames (slow)
    total_frames = int(_info.get('nb_frames', 0) or 0)
    if total_frames <= 0:
        print("Counting frames (packet scan)...", flush=True)
        _probe2 = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
             '-show_entries', 'stream=nb_read_packets', '-of', 'json', input_video],
            capture_output=True, text=True)
        _info2 = _json.loads(_probe2.stdout)['streams'][0]
        total_frames = int(_info2.get('nb_read_packets', 0) or 0)
    if total_frames <= 0:
        print("Counting frames (full decode — this may take a while)...", flush=True)
        _probe3 = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
             '-show_entries', 'stream=nb_read_frames', '-of', 'json', input_video],
            capture_output=True, text=True)
        _info3 = _json.loads(_probe3.stdout)['streams'][0]
        total_frames = int(_info3.get('nb_read_frames', 0) or 0)

    with open(os.path.join(frames_dir, 'fps.txt'), 'w') as fh:
        fh.write(f"{fps:.6f}")

    # Auto-probe optimal window size if requested
    if window_size < 0:
        window_size = probe_window_size(model, 15, src_h, src_w, device, use_half)
    print(f"Window size: {window_size}", flush=True)

    # ── Sliding-window parameters ────────────────────────────────────────────
    OVERLAP = min(2, window_size // 4)
    stride  = max(1, window_size - OVERLAP)

    # Resume: find last completed frame and calculate where to restart
    resume_from = 0
    if resume:
        existing = sorted([f for f in os.listdir(frames_dir)
                           if f.startswith('frame_') and f.endswith('.png')])
        if existing:
            last_idx = int(existing[-1].replace('frame_', '').replace('.png', ''))
            resume_from = last_idx + 1
            if resume_from >= max(total_frames, 1):
                print(f"Resume: all {len(existing)} frames already present — skipping upscale")
                return True
            # Snap to a window boundary so we produce contiguous output.
            # First window outputs window_size frames, subsequent output stride each.
            if resume_from <= window_size:
                resume_from = 0  # still in first window, restart from beginning
            else:
                resume_from = window_size + ((resume_from - window_size) // stride) * stride
            print(f"Resume: restarting from output frame {resume_from}")

    # ── FFmpeg pipe reader (reliable for all containers, unlike OpenCV) ────────
    ffmpeg_cmd = [
        'ffmpeg', '-v', 'error', '-i', input_video,
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE,
                                   bufsize=src_w * src_h * 3 * 4)
    frame_nbytes = src_w * src_h * 3

    # If resuming, skip source frames to reach the restart point.
    # Source frames consumed = resume_from (first window_size, then stride each).
    # We need OVERLAP extra context frames before the window, so back up slightly.
    if resume_from > 0:
        if resume_from <= window_size:
            skip_source = 0
        else:
            skip_source = resume_from - OVERLAP
        if skip_source > 0:
            print(f"Resume: skipping {skip_source} source frames...", flush=True)
            for _ in range(skip_source):
                raw = ffmpeg_proc.stdout.read(frame_nbytes)
                if len(raw) < frame_nbytes:
                    break

    print(f"Temporal window: {window_size}  |  Overlap: {OVERLAP}  |  Stride: {stride}")
    print(f"Output: {output_w}×{output_h}")

    def read_n(n):
        frames = []
        for _ in range(n):
            raw = ffmpeg_proc.stdout.read(frame_nbytes)
            if len(raw) < frame_nbytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(src_h, src_w, 3)
            frames.append(frame)
        return frames

    # Prime: read first full window
    window = read_n(window_size)
    if not window:
        ffmpeg_proc.stdout.close()
        ffmpeg_proc.wait()
        return False

    eof = len(window) < window_size
    # Pad a short first window with copies of the first frame
    while len(window) < window_size:
        window.insert(0, window[0])

    output_idx = resume_from
    is_first   = (resume_from == 0)

    with tqdm(total=total_frames if total_frames > 0 else None,
              initial=resume_from,
              desc="Upscaling frames (temporal)", unit="frame") as pbar:
        while True:
            # Build [1, T, C, H, W] batch
            tensors = torch.stack(
                [frame_to_tensor(f, device, use_half) for f in window], dim=0
            ).unsqueeze(0)

            with torch.no_grad():
                out_batch = model(tensors)  # [1, T, C, 4H, 4W]

            # Which frames from this window are "valid" output?
            # Left trim only — skip OVERLAP frames already output by the previous window.
            # No right trim: BasicVSR++ is bidirectional so edge quality is fine.
            valid_start = 0 if is_first else OVERLAP
            valid_end   = len(window)

            for j in range(valid_start, valid_end):
                out_frame = tensor_to_frame(out_batch[0, j])

                src_h = window[j].shape[0]
                if out_frame.shape[0] != output_h or out_frame.shape[1] != output_w:
                    interp = (cv2.INTER_AREA if output_h < src_h * MODEL_SCALE
                              else cv2.INTER_LANCZOS4)
                    out_frame = cv2.resize(out_frame, (output_w, output_h),
                                           interpolation=interp)

                frame_path = os.path.join(frames_dir, f'frame_{output_idx:08d}.png')
                cv2.imwrite(frame_path, out_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                output_idx += 1
                pbar.update(1)

            # Free GPU memory from this iteration before building the next window
            del tensors, out_batch
            torch.cuda.empty_cache()

            if eof:
                break

            is_first = False

            # Slide: keep OVERLAP trailing frames as leading context for next window
            context    = window[-OVERLAP:] if OVERLAP > 0 else []
            new_frames = read_n(stride)
            if len(new_frames) < stride:
                eof = True
            window = context + new_frames

    ffmpeg_proc.stdout.close()
    ffmpeg_proc.wait()

    if total_frames > 0 and output_idx < total_frames:
        print(f"WARNING: produced {output_idx}/{total_frames} frames "
              f"({total_frames - output_idx} missing)")

    return output_idx > 0


if __name__ == '__main__':
    _, input_video, frames_dir, model_key, model_path, spynet_path, \
        window_size, output_w, output_h, \
        use_full_precision_str, resume_str = sys.argv

    success = upscale_temporal(
        input_video        = input_video,
        frames_dir         = frames_dir,
        model_key          = model_key,
        model_path         = model_path,
        spynet_path        = spynet_path,
        window_size        = -1 if window_size == 'auto' else int(window_size),
        output_w           = int(output_w),
        output_h           = int(output_h),
        use_full_precision = (use_full_precision_str == 'true'),
        resume             = (resume_str == 'true'),
    )
    sys.exit(0 if success else 1)
TEMPORAL_EOF
}

upscale_video() {
    local output_file="$1"
    local frames_dir="$TEMP_DIR/frames"

    # SeedVR2 runs in its own venv and produces a finished video, not PNG frames.
    if [[ "$IS_SEEDVR2" == true ]]; then
        upscale_seedvr2 "$output_file"
        return
    fi

    mkdir -p "$frames_dir"
    source "$VENV_DIR/bin/activate"

    if [[ "$IS_TEMPORAL" == true ]]; then
        # Temporal pipeline — basicsr multi-frame models
        if ! "$VENV_DIR/bin/python3" -c "import importlib.util; exit(0 if importlib.util.find_spec('basicsr') else 1)" &>/dev/null; then
            print_error "basicsr not installed — required for temporal models (basicvsr, realbasicvsr)"
            print_error "Fix: source $VENV_DIR/bin/activate && pip install basicsr"
            exit 1
        fi
        write_temporal_python_script
        print_info "Starting temporal AI upscaling..."
        local win_note="$TEMPORAL_WINDOW"
        [[ "$TEMPORAL_WINDOW" == "auto" ]] && win_note="auto (probing GPU)"
        print_info "Model: $MODEL_KEY  |  Window: ${win_note}  |  Prefilter: ${PREFILTER}"

        "$VENV_DIR/bin/python3" "$TEMP_DIR/upscale_temporal.py" \
            "$UPSCALE_SOURCE" \
            "$frames_dir" \
            "$MODEL_KEY" \
            "$MODEL_PATH" \
            "${SPYNET_PATH:-}" \
            "$TEMPORAL_WINDOW" \
            "$OUTPUT_WIDTH" \
            "$OUTPUT_HEIGHT" \
            "$FULL_PRECISION" \
            "$RESUME"
    else
        # Single-frame pipeline — spandrel models
        write_python_script
        local tile_note="$TILE_SIZE"
        [[ "$TILE_SIZE" == "auto" ]] && tile_note="auto (probing GPU)"
        [[ "$TILE_SIZE" == "0" ]] && tile_note="0 (full-frame)"
        print_info "Starting AI upscaling..."
        print_info "Model: $MODEL_KEY  |  Tile: ${tile_note}  |  Tile-pad: ${TILE_PAD}  |  Prefilter: ${PREFILTER}"

        "$VENV_DIR/bin/python3" "$TEMP_DIR/upscale.py" \
            "$UPSCALE_SOURCE" \
            "$frames_dir" \
            "$MODEL_PATH" \
            "$TILE_SIZE" \
            "$TILE_PAD" \
            "$OUTPUT_WIDTH" \
            "$OUTPUT_HEIGHT" \
            "$FULL_PRECISION" \
            "$RESUME"
    fi

    print_success "Upscaling complete"
    encode_output "$output_file"
}

# Core SeedVR2 inference: one input video → one upscaled video. Used for both the single-shot
# and per-segment paths. Resolves the lossless speed options and runs the standalone CLI.
seedvr2_infer() {
    local in_video="$1" out_video="$2"

    # Lossless speedups (no quality change): torch.compile + chosen attention backend.
    # 'auto' picks flash_attn_2 when flash-attn is importable in the SeedVR2 venv, else sdpa.
    local attn="$SEEDVR2_ATTENTION"
    if [[ "$attn" == "auto" ]]; then
        if "$SEEDVR2_VENV/bin/python3" -c "import flash_attn" &>/dev/null; then
            attn="flash_attn_2"
        else
            attn="sdpa"
        fi
    fi
    local -a speed_opts=(--attention_mode "$attn")
    [[ "$SEEDVR2_COMPILE" == true ]] && speed_opts+=(--compile_dit)
    [[ "$SEEDVR2_COMPILE_VAE" == true ]] && speed_opts+=(--compile_vae)

    mkdir -p "$SEEDVR2_MODEL_DIR"
    "$SEEDVR2_VENV/bin/python3" "$SEEDVR2_CLI" \
        "$in_video" \
        --output "$out_video" \
        --output_format mp4 \
        --video_backend ffmpeg \
        --10bit \
        --model_dir "$SEEDVR2_MODEL_DIR" \
        --dit_model "$SEEDVR2_MODEL_FILE" \
        --resolution "$OUTPUT_HEIGHT" \
        --batch_size "$SEEDVR2_BATCH" \
        --chunk_size "$SEEDVR2_CHUNK" \
        --temporal_overlap "$SEEDVR2_TEMPORAL_OVERLAP" \
        --color_correction lab \
        --blocks_to_swap "$SEEDVR2_BLOCKS_SWAP" \
        --dit_offload_device cpu \
        --vae_offload_device cpu \
        --vae_encode_tiled --vae_encode_tile_size "$SEEDVR2_VAE_ENC_TILE" \
        --vae_decode_tiled --vae_decode_tile_size "$SEEDVR2_VAE_DEC_TILE" \
        "${speed_opts[@]}"
}

upscale_seedvr2() {
    local output_file="$1"
    local raw_video="$TEMP_DIR/seedvr2_raw.mkv"
    local feed="$UPSCALE_SOURCE"

    # SeedVR2 has no aspect awareness — it upscales stored pixels as-if-square. For
    # anamorphic sources (e.g. PAL 720x576 shown 16:9) that distorts the result. We
    # de-anamorphize UP FRONT so the model reconstructs detail at correct geometry,
    # rather than stretching (and softening) it afterwards. The scale 'dar' variable
    # is the source's display aspect — for square-pixel sources this is a no-op.
    if [[ "$INPUT_SAR" != "1:1" && -n "$INPUT_SAR" && "$INPUT_SAR" != "N/A" ]]; then
        local square="$TEMP_DIR/seedvr2_square.mkv"
        if [[ "$RESUME" == true && -f "$square" ]]; then
            print_info "Resume: reusing de-anamorphized source"
            feed="$square"
        else
            print_info "SeedVR2: de-anamorphizing source to square pixels (SAR ${INPUT_SAR})"
            ffmpeg -i "$feed" \
                -vf "scale=w='trunc(ih*dar/2)*2':h=ih:flags=lanczos,setsar=1" \
                -an -sn -c:v libx264 -qp 0 -pix_fmt yuv420p \
                "$square" -y -loglevel error \
                && feed="$square" \
                || print_warning "De-anamorphize failed — feeding original; final encode still corrects AR"
        fi
    fi

    print_info "Model: seedvr2 (${SEEDVR2_MODEL_FILE})  |  short-edge: ${OUTPUT_HEIGHT}px  |  batch: ${SEEDVR2_BATCH}  |  chunk: ${SEEDVR2_CHUNK}  |  Prefilter: ${PREFILTER}"
    print_warning "Diffusion VSR is slow (~4-7 s/frame on 16GB)."
    print_warning "First run downloads weights (~3.6GB) to ${SEEDVR2_MODEL_DIR}."

    # Long files → segment for resumability (a multi-day single run can't survive any interruption).
    local dur_int=${DURATION%.*}
    if [[ "$SEEDVR2_SEGMENT_SECONDS" -gt 0 && "${dur_int:-0}" -gt "$SEEDVR2_SEGMENT_SECONDS" ]]; then
        seedvr2_segmented "$feed" "$output_file"
        return
    fi

    # Single-shot (short files).
    if [[ "$RESUME" == true && -f "$raw_video" ]]; then
        print_info "Resume: reusing existing SeedVR2 output ($raw_video)"
    else
        print_info "Starting SeedVR2 diffusion upscaling..."
        seedvr2_infer "$feed" "$raw_video"
        [[ -f "$raw_video" ]] || { print_error "SeedVR2 produced no output — see errors above."; exit 1; }
    fi

    print_success "Upscaling complete"
    # Reuse the shared encoder. Aspect is already correct (de-anamorphized up front), so this
    # stream-COPIES SeedVR2's video and just muxes audio/subs — no re-encode, no quality loss
    # (unless --sharpen forces one). Passing the video as $2 selects that path.
    encode_output "$output_file" "$raw_video"
}

# Auto-segmented SeedVR2 for long files: split → upscale each segment atomically → concat → mux.
# Each segment's output is written to a .part file and only renamed on success, so an interrupted
# run never leaves a half-done segment that --resume would wrongly skip.
seedvr2_segmented() {
    local feed="$1" output_file="$2"
    local seg_dir="$TEMP_DIR/seedvr2_segments"
    local concat_raw="$TEMP_DIR/seedvr2_concat.mkv"
    mkdir -p "$seg_dir"

    # 1. Split the feed into video-only segments (lossless stream copy; cuts snap to keyframes —
    #    the prefilter's FFV1 / de-anamorph intermediates are all-intra, so cuts are exact).
    #    Deterministic boundaries mean --resume re-derives the identical segment set.
    if [[ "$RESUME" == true && -n "$(ls "$seg_dir"/in_*.mkv 2>/dev/null)" ]]; then
        print_info "Resume: reusing existing input segments"
    else
        rm -f "$seg_dir"/in_*.mkv
        print_info "Segmenting source into ${SEEDVR2_SEGMENT_SECONDS}s pieces..."
        ffmpeg -i "$feed" -map 0:v -c copy -f segment \
            -segment_time "$SEEDVR2_SEGMENT_SECONDS" -reset_timestamps 1 \
            "$seg_dir/in_%04d.mkv" -y -loglevel error
    fi

    local -a segs=("$seg_dir"/in_*.mkv)
    local total=${#segs[@]}
    [[ -f "${segs[0]}" ]] || { print_error "Segmentation produced no segments"; exit 1; }
    print_info "SeedVR2: ${total} segments of ~${SEEDVR2_SEGMENT_SECONDS}s each (auto-segmented for resumability)"
    print_warning "Long SeedVR2 run: expect DAYS of compute (~4-7 s/frame). It is fully resumable —"
    print_warning "if interrupted, re-run the SAME command with --resume to skip finished segments."
    print_warning "Keep plenty of free disk for temp segments; --prefilter none avoids a large FFV1 intermediate."

    # 2. Upscale each segment (skip ones already finished on a --resume).
    local i=0 done_count=0
    for seg in "${segs[@]}"; do
        i=$((i + 1))
        local idx; idx=$(basename "$seg" .mkv); idx=${idx#in_}
        local out="$seg_dir/out_${idx}.mkv"
        local part="$seg_dir/part_${idx}.mkv"   # valid .mkv ext (ffmpeg needs it); distinct from out_*
        if [[ "$RESUME" == true && -f "$out" ]]; then
            print_info "Segment ${i}/${total}: already done — skipping"
            done_count=$((done_count + 1))
            continue
        fi
        print_info "Segment ${i}/${total}: upscaling $(basename "$seg")..."
        seedvr2_infer "$seg" "$part"
        [[ -f "$part" ]] || { print_error "Segment ${i} produced no output — see errors above."; exit 1; }
        mv -f "$part" "$out"   # atomic: only a complete segment counts as done
        done_count=$((done_count + 1))
    done

    # 3. Concat the upscaled segments losslessly (all share SeedVR2's HEVC params).
    print_info "Concatenating ${done_count} upscaled segments..."
    local list="$seg_dir/concat.txt"
    : > "$list"
    for seg in "${segs[@]}"; do
        local idx; idx=$(basename "$seg" .mkv); idx=${idx#in_}
        printf "file '%s'\n" "$seg_dir/out_${idx}.mkv" >> "$list"
    done
    ffmpeg -f concat -safe 0 -i "$list" -c copy "$concat_raw" -y -loglevel error \
        || { print_error "Concat failed"; exit 1; }

    print_success "Upscaling complete"
    # Mux audio/subs onto the concatenated video — stream copy, no re-encode.
    encode_output "$output_file" "$concat_raw"
}

encode_output() {
    local output_file="$1"
    local video_input="${2:-}"   # if set, encode from this video; else from frame PNGs in TEMP_DIR/frames
    local frames_dir="$TEMP_DIR/frames"

    local crf
    case "$QUALITY" in
        high)   crf=16 ;;
        medium) crf=20 ;;
        low)    crf=24 ;;
        *)      print_error "Unknown quality: $QUALITY"; exit 1 ;;
    esac

    local x265_preset="$ENCODE_SPEED"
    case "$ENCODE_SPEED" in
        slow|medium|fast) ;;
        *)  print_error "Unknown encode speed: $ENCODE_SPEED"; exit 1 ;;
    esac

    # A finished video (SeedVR2) whose pixels are already final + correct AR can be stream-COPIED
    # with audio/subs muxed on — no second encode, no generation loss. We only need to re-encode
    # it if --sharpen was requested (a filter can't apply to a copied stream). The PNG-frame path
    # (spandrel/basicsr) always encodes once here — that single pass is the deliverable encode.
    local lossless_copy=false
    if [[ -n "$video_input" && "$SHARPEN" == false ]]; then
        lossless_copy=true
    fi

    # Primary input: finished video, or the upscaled PNG sequence (PNGs carry no fps → fps.txt).
    local -a primary_input=()
    local fps_note=""
    if [[ -n "$video_input" ]]; then
        primary_input=(-i "$video_input")
    else
        local fps; fps=$(cat "$frames_dir/fps.txt")
        primary_input=(-framerate "$fps" -pattern_type glob -i "$frames_dir/frame_*.png")
        fps_note=" @ ${fps}fps"
    fi

    # Build ffmpeg inputs and mapping for audio/subtitles (extracted from the original).
    local -a extra_inputs=()
    local -a extra_maps=()
    local -a extra_codecs=()

    # Track the ffmpeg input index as we add streams (0 is the primary video) so the -map
    # indices stay correct no matter which of audio/subs are included. Each extracted stream
    # is validated with probe_ok — a corrupt audio.mka/subs.mkv is skipped (with a warning)
    # rather than failing the whole encode. (Bad subs can happen when a clip is cut mid-stream.)
    local input_idx=1
    if [[ "$HAS_AUDIO" == true && -f "$TEMP_DIR/audio.mka" ]]; then
        if probe_ok "$TEMP_DIR/audio.mka"; then
            extra_inputs+=(-i "$TEMP_DIR/audio.mka")
            extra_maps+=(-map "${input_idx}:a")
            extra_codecs+=(-c:a copy)
            input_idx=$((input_idx + 1))
        else
            print_warning "Extracted audio is unreadable — encoding without audio"
        fi
    fi
    if [[ "$HAS_SUBS" == true && -f "$TEMP_DIR/subs.mkv" ]]; then
        if probe_ok "$TEMP_DIR/subs.mkv"; then
            extra_inputs+=(-i "$TEMP_DIR/subs.mkv")
            extra_maps+=(-map "${input_idx}:s")
            extra_codecs+=(-c:s copy)
            input_idx=$((input_idx + 1))
        else
            print_warning "Extracted subtitles are unreadable — encoding without subtitles"
        fi
    fi

    if [[ "$lossless_copy" == true ]]; then
        print_info "Muxing (lossless stream copy — no re-encode): → $output_file"
        ffmpeg \
            "${primary_input[@]}" \
            "${extra_inputs[@]}" \
            -map 0:v \
            "${extra_maps[@]}" \
            -c:v copy \
            "${extra_codecs[@]}" \
            -movflags +faststart \
            "$output_file" -y -loglevel error -stats
    else
        print_info "Encoding: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}${fps_note}  preset:${ENCODE_SPEED} crf:${crf} → $output_file"
        # Final resize to exact target dimensions (safety net for rounding differences) + sharpen.
        local vf_out="scale=${OUTPUT_WIDTH}:${OUTPUT_HEIGHT}:flags=lanczos"
        if [[ "$SHARPEN" == true ]]; then
            vf_out="${vf_out},unsharp=3:3:0.5:3:3:0.0"
        fi
        ffmpeg \
            "${primary_input[@]}" \
            "${extra_inputs[@]}" \
            -map 0:v \
            "${extra_maps[@]}" \
            -vf "$vf_out" \
            -c:v libx265 -crf "$crf" -preset "$x265_preset" \
            -pix_fmt yuv420p10le \
            -x265-params "no-open-gop=1:keyint=250:bframes=8:aq-mode=3" \
            "${extra_codecs[@]}" \
            -movflags +faststart \
            "$output_file" -y -loglevel error -stats
    fi

    print_success "Output saved: $output_file"
}

simple_scale() {
    local input_file="$1"
    local output_file="$2"
    print_info "Source >= target — downscaling with FFmpeg (no AI needed)..."

    local crf
    case "$QUALITY" in
        high)   crf=16 ;;
        medium) crf=20 ;;
        low)    crf=24 ;;
        *)      print_error "Unknown quality: $QUALITY"; exit 1 ;;
    esac

    local -a extra_codecs=(-c:a copy)
    if [[ "$HAS_SUBS" == true ]]; then
        extra_codecs+=(-c:s copy)
    fi

    ffmpeg -i "$input_file" \
        -map 0 \
        -vf "scale=${OUTPUT_WIDTH}:${OUTPUT_HEIGHT}:flags=lanczos" \
        -c:v libx265 -crf "$crf" -preset "$ENCODE_SPEED" \
        -pix_fmt yuv420p10le \
        "${extra_codecs[@]}" \
        -movflags +faststart \
        "$output_file" -y -loglevel error -stats

    print_success "Output saved: $output_file"
}

cleanup() {
    if [[ "$KEEP_TEMP" == false ]]; then
        print_info "Cleaning up temp files..."
        find "$TEMP_DIR" -type f -delete 2>/dev/null || true
        rm -rf "$TEMP_DIR" 2>/dev/null || true
        print_success "Cleanup done"
    else
        print_info "Temp files kept in: $TEMP_DIR"
    fi
}

##############################################################################
# Argument parsing
##############################################################################

INPUT_FILE=""
OUTPUT_FILE=""
RESOLUTION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)          INPUT_FILE="$2";      shift 2 ;;
        -o|--output)         OUTPUT_FILE="$2";     shift 2 ;;
        -r|--resolution)     RESOLUTION="$2";      shift 2 ;;
        -m|--model)          MODEL_KEY="$2";       shift 2 ;;
        -t|--tile)           TILE_SIZE="$2"; TILE_SIZE_EXPLICIT=true; shift 2 ;;
        --tile-pad)          TILE_PAD="$2";        shift 2 ;;
        --temporal-window)   TEMPORAL_WINDOW="$2"; shift 2 ;;
        -q|--quality)        QUALITY="$2";         shift 2 ;;
        --encode-speed)      ENCODE_SPEED="$2";    shift 2 ;;
        --prefilter)         PREFILTER="$2"; PREFILTER_EXPLICIT=true; shift 2 ;;
        --deinterlace)       DEINTERLACE=true;     shift   ;;
        --resume)            RESUME=true;          shift   ;;
        --sharpen)           SHARPEN=true;         shift   ;;
        --full-precision)    FULL_PRECISION=true;  shift   ;;
        --keep-temp)         KEEP_TEMP=true;       shift   ;;
        -h|--help)        usage ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

##############################################################################
# Interactive mode — triggered when only -i is provided (no -r)
##############################################################################

interactive_setup() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              AI Video Upscaler — Interactive Setup          ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # ── Resolution ────────────────────────────────────────────────────────────
    echo -e "${BLUE}Target resolution:${NC}"
    echo "  1) 720p   — SD → HD"
    echo "  2) 1080p  — SD/HD → Full HD"
    echo "  3) 1440p  — HD → 2K"
    echo "  4) 2160p  — HD/FHD → 4K"
    echo ""
    while true; do
        read -rp "Choose [1-4]: " res_choice
        case "$res_choice" in
            1) RESOLUTION="720p";  break ;;
            2) RESOLUTION="1080p"; break ;;
            3) RESOLUTION="1440p"; break ;;
            4) RESOLUTION="2160p"; break ;;
            *) echo "  Invalid choice, try again." ;;
        esac
    done
    echo ""

    # ── Model ─────────────────────────────────────────────────────────────────
    echo -e "${BLUE}AI model:${NC}"
    echo ""
    echo -e "  ${GREEN}Single-frame models${NC} (process each frame independently)"
    echo ""
    echo -e "  ${YELLOW}SPAN — fastest (~0.5-1 s/frame):${NC}"
    echo "   1) spanmedium   — Nomos8k + real-world degradation, medium [best fast option]"
    echo "   2) spanweak     — Nomos8k, light degradation (cleaner/better sources)"
    echo "   3) spanstrong   — Nomos8k, heavy degradation (badly compressed sources)"
    echo ""
    echo -e "  ${YELLOW}RealPLKSR — fast (~1-2 s/frame):${NC}"
    echo "   4) webphoto     — Web/streaming sources: lens blur + JPEG/WebP + noise"
    echo "   5) nomos2plksr  — Cleaner compressed sources (JPEG only, less aggressive)"
    echo ""
    echo -e "  ${YELLOW}RRDB — standard (~4 s/frame):${NC}"
    echo "   6) nomos8k      — Best all-rounder for compressed live-action [default]"
    echo "   7) lsdirplus    — LSDIR dataset + real degradation, sharp detail on degraded sources"
    echo "   8) lsdir        — LSDIR dataset, sharp detail on clean sources"
    echo "   9) ultrasharp   — Maximum sharpness on clean sources (Blu-ray)"
    echo "  10) realesrgan   — Legacy fallback"
    echo ""
    echo -e "  ${YELLOW}Transformer — slow (20-60 s/frame, short clips only):${NC}"
    echo "  11) atdjpg       — Best for heavily JPEG-compressed/degraded sources"
    echo "  12) nomos8kschat — HAT-L on Nomos8k, real-world/compressed sources"
    echo "  13) hat          — Highest fidelity, clean sources only"
    echo "  14) nomos8kdat   — DAT transformer, highest quality"
    echo ""
    echo -e "  ${GREEN}Temporal models${NC} (multi-frame — best consistency, much faster for long content)"
    echo "  15) basicvsr     — Best for TV/movies, degraded or compressed sources"
    echo ""
    echo -e "  ${GREEN}Diffusion VSR${NC} (highest quality on low-res sources — but slow, short clips)"
    echo "  16) seedvr2      — Reconstructs detail; biggest jump on low-res/compressed (~4-7 s/frame)"
    echo ""
    while true; do
        read -rp "Choose [1-16, default=1]: " model_choice
        local selected_key=""
        case "${model_choice:-1}" in
            1)  selected_key="spanmedium"   ;;
            2)  selected_key="spanweak"     ;;
            3)  selected_key="spanstrong"   ;;
            4)  selected_key="webphoto"     ;;
            5)  selected_key="nomos2plksr"  ;;
            6)  selected_key="nomos8k"      ;;
            7)  selected_key="lsdirplus"    ;;
            8)  selected_key="lsdir"        ;;
            9)  selected_key="ultrasharp"   ;;
            10) selected_key="realesrgan"   ;;
            11) selected_key="atdjpg"       ;;
            12) selected_key="nomos8kschat" ;;
            13) selected_key="hat"          ;;
            14) selected_key="nomos8kdat"   ;;
            15) selected_key="basicvsr"     ;;
            16) selected_key="seedvr2"      ;;
            *) echo "  Invalid choice, try again."; continue ;;
        esac
        # Validate availability. SeedVR2 lives in its own venv (no .pth); check the CLI.
        if [[ "$selected_key" == "seedvr2" ]]; then
            if [[ ! -f "$SEEDVR2_CLI" || ! -x "$SEEDVR2_VENV/bin/python3" ]]; then
                echo -e "  ${RED}SeedVR2 not installed.${NC} Run: prototype/seedvr2/setup.sh"
                continue
            fi
        else
            local model_file=""
            if [[ -n "${MODEL_FILES[$selected_key]+_}" ]]; then
                model_file="$MODEL_DIR/${MODEL_FILES[$selected_key]}"
            elif [[ -n "${TEMPORAL_MODEL_FILES[$selected_key]+_}" ]]; then
                model_file="$MODEL_DIR/${TEMPORAL_MODEL_FILES[$selected_key]}"
            fi
            if [[ ! -f "$model_file" ]]; then
                echo -e "  ${RED}Model not found:${NC} $model_file"
                echo "  Download it to $MODEL_DIR and try again, or choose a different model."
                continue
            fi
        fi
        MODEL_KEY="$selected_key"
        break
    done
    echo ""

    # ── Prefilter ─────────────────────────────────────────────────────────────
    # SeedVR2 (generative, trained on degradation) does best with no pre-denoise, so its
    # default is 'none'; the discriminative models default to 'light'.
    local pf_default=2
    [[ "$MODEL_KEY" == "seedvr2" ]] && pf_default=1
    echo -e "${BLUE}Pre-filter (noise/artifact reduction before AI upscaling):${NC}"
    if [[ "$MODEL_KEY" == "seedvr2" ]]; then
        echo "  1) none    — Recommended for SeedVR2 (it reconstructs from real degradation) [default]"
        echo "  2) light   — Mild denoise"
    else
        echo "  1) none    — Clean sources: Blu-ray, high-quality 1080p"
        echo "  2) light   — Good default: mild denoise, safe for most content [default]"
    fi
    echo "  3) medium  — Compressed/blocky sources: web downloads, streaming rips"
    echo "  4) heavy   — Badly degraded: VHS, old TV recordings, heavy compression"
    echo ""
    while true; do
        read -rp "Choose [1-4, default=${pf_default}]: " pf_choice
        case "${pf_choice:-$pf_default}" in
            1) PREFILTER="none";   break ;;
            2) PREFILTER="light";  break ;;
            3) PREFILTER="medium"; break ;;
            4) PREFILTER="heavy";  break ;;
            *) echo "  Invalid choice, try again." ;;
        esac
    done
    echo ""

    # ── Deinterlace ───────────────────────────────────────────────────────────
    echo -e "${BLUE}Deinterlace?${NC}"
    echo "  Recommended for interlaced sources (DVD, old TV broadcasts, camcorder footage)."
    echo "  Safe to enable on progressive sources — it will auto-detect and preserve framerate."
    echo ""
    while true; do
        read -rp "Enable deinterlace? [y/N]: " di_choice
        case "${di_choice:-n}" in
            [yY]|[yY][eE][sS]) DEINTERLACE=true;  break ;;
            [nN]|[nN][oO]|"")  DEINTERLACE=false; break ;;
            *) echo "  Invalid choice, try again." ;;
        esac
    done
    echo ""

    # ── Encode speed ──────────────────────────────────────────────────────────
    echo -e "${BLUE}Encode speed${NC} (x265 preset — affects final encode time, not AI upscaling):"
    echo "  1) slow     Best quality, slowest encode (default)"
    echo "  2) medium   Balanced — ~2x faster than slow"
    echo "  3) fast     Quick encode — ~4x faster, slightly lower quality"
    echo ""
    while true; do
        read -rp "Choose [1-3, default=1]: " es_choice
        case "${es_choice:-1}" in
            1) ENCODE_SPEED="slow";   break ;;
            2) ENCODE_SPEED="medium"; break ;;
            3) ENCODE_SPEED="fast";   break ;;
            *) echo "  Invalid choice, try again." ;;
        esac
    done
    echo ""

    # ── Summary ───────────────────────────────────────────────────────────────
    echo -e "${GREEN}────────────────────────────────────────────────────────────────${NC}"
    echo -e "  Input:        $INPUT_FILE"
    echo -e "  Resolution:   $RESOLUTION"
    echo -e "  Model:        $MODEL_KEY"
    echo -e "  Prefilter:    $PREFILTER"
    echo -e "  Deinterlace:  $DEINTERLACE"
    echo -e "  Encode speed: $ENCODE_SPEED"
    echo -e "${GREEN}────────────────────────────────────────────────────────────────${NC}"
    echo ""
    while true; do
        read -rp "Proceed? [Y/n]: " confirm
        case "${confirm:-y}" in
            [yY]|[yY][eE][sS]|"") break ;;
            [nN]|[nN][oO])
                echo "Aborted."
                exit 0
                ;;
            *) echo "  Invalid choice, try again." ;;
        esac
    done
    echo ""
}

##############################################################################
# Main
##############################################################################

if [[ -z "$INPUT_FILE" ]]; then
    print_error "Missing input file (-i)"
    usage
fi

if [[ -z "$RESOLUTION" ]]; then
    interactive_setup
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    print_error "Input file not found: $INPUT_FILE"
    exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    BASENAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')
    OUTPUT_FILE="${BASENAME}_upscaled_${RESOLUTION}.mkv"
fi

mkdir -p "$TEMP_DIR"

if [[ "$RESUME" == false ]]; then
    # Fresh run — clear previous temp data
    find "$TEMP_DIR/frames" -type f -delete 2>/dev/null || true
    rm -rf "$TEMP_DIR/frames"
    rm -f  "$TEMP_DIR/cleaned_source.mkv"
    rm -f  "$TEMP_DIR/upscale.py"
    rm -f  "$TEMP_DIR/upscale_temporal.py"
    rm -f  "$TEMP_DIR/audio.mka"
    rm -f  "$TEMP_DIR/subs.mkv"
    rm -f  "$TEMP_DIR/seedvr2_raw.mkv"
    rm -f  "$TEMP_DIR/seedvr2_square.mkv"
    rm -f  "$TEMP_DIR/seedvr2_concat.mkv"
    rm -rf "$TEMP_DIR/seedvr2_segments"
fi

mkdir -p "$TEMP_DIR/frames"

print_info "=== AI Video Upscaler ==="
print_info "Input:  $INPUT_FILE"
print_info "Target: $RESOLUTION → $OUTPUT_FILE"
print_info ""

check_dependencies
get_video_info "$INPUT_FILE"
calculate_scale "$RESOLUTION"

if [[ "$USE_AI" == true ]]; then
    select_model

    # Transformer models overflow float16 — auto-enable full precision
    case "$MODEL_KEY" in
    hat|nomos8kdat|atdjpg|nomos8kschat)
        if [[ "$FULL_PRECISION" == false ]]; then
            FULL_PRECISION=true
            print_info "Auto-enabling --full-precision for $MODEL_KEY (transformer models overflow float16)"
        fi
        ;;
    esac

    # SeedVR2 is generative and trained on degraded input, so pre-denoising tends to remove detail
    # cues it would otherwise reconstruct. Default it to no prefilter unless the user asked for one.
    if [[ "$IS_SEEDVR2" == true && "$PREFILTER_EXPLICIT" == false ]]; then
        PREFILTER="none"
        print_info "SeedVR2: defaulting to --prefilter none (diffusion handles degradation; pass --prefilter to override)"
    fi

    # Tile size: if user didn't pass -t, Python will auto-probe the optimal size.
    # If user passed -t explicitly, that value is used as-is.

    extract_audio "$INPUT_FILE"
    extract_subs "$INPUT_FILE"
    run_prefilter "$INPUT_FILE"
    upscale_video "$OUTPUT_FILE"
else
    simple_scale "$INPUT_FILE" "$OUTPUT_FILE"
fi

cleanup

print_success "=== Complete ==="
OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" 2>/dev/null | cut -f1 || echo "unknown")
print_success "Output: $OUTPUT_FILE ($OUTPUT_SIZE)"
