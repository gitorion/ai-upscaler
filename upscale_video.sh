#!/bin/bash

##############################################################################
# AI Video Upscaler — spandrel + basicsr edition
# Single-frame models: nomos8k (default), nomos8kdat, lsdir, ultrasharp, realesrgan, hat
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
TEMP_DIR="$SCRIPT_DIR/temp"
VENV_DIR="$SCRIPT_DIR/venv"

# ── Defaults ──────────────────────────────────────────────────────────────────
TILE_SIZE=512
TILE_SIZE_EXPLICIT=false
TILE_PAD=64
MODEL_KEY="nomos8k"
QUALITY="high"
PREFILTER="light"
DEINTERLACE=false
RESUME=false
SHARPEN=false
FULL_PRECISION=false
KEEP_TEMP=false
UPSCALE_SOURCE=""
IS_TEMPORAL=false
TEMPORAL_WINDOW=15
SPYNET_PATH=""

# ── Single-frame model registry (spandrel) — all 4x ──────────────────────────
declare -A MODEL_FILES=(
    [nomos8k]="4xNomos8kSC.pth"
    [nomos8kdat]="4xNomos8kDAT.pth"
    [lsdir]="4xLSDIR.pth"
    [ultrasharp]="4x-UltraSharp.pth"
    [realesrgan]="RealESRGAN_x4plus.pth"
    [hat]="HAT-L_SRx4_ImageNet-pretrain.pth"
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

usage() {
    cat << EOF
${GREEN}AI Video Upscaler${NC} — spandrel + basicsr edition, live-action optimised

Usage: $0 -i INPUT -r RESOLUTION [OPTIONS]

Required:
  -i, --input FILE          Input video file
  -r, --resolution RES      Target resolution: 720p, 1080p, 1440p, 2160p

Model selection (-m / --model):
  ── Single-frame models (spandrel) ──────────────────────────────────────────
  nomos8k     Best all-round for compressed live-action — fast (~4s/frame)  ← default
  nomos8kdat  DAT transformer — highest single-frame quality, ~6× slower (~22s/frame)
  lsdir       Sharp detail, handles real-world degradations
  ultrasharp  Maximum sharpness (better on cleaner sources)
  realesrgan  Real-ESRGAN x4plus (legacy fallback)
  hat         HAT-L — highest fidelity single-frame, clean sources only

  ── Temporal models (basicsr) — multi-frame, best temporal consistency ─────
  basicvsr    BasicVSR++ — strong on real-world degraded video  (requires basicsr)

  Temporal models process a sliding window of frames simultaneously using optical
  flow, producing sharper results with far less frame-to-frame flickering.
  Requires: pip install basicsr  +  spynet_20210409-c6c1bd09.pth in models/
  See --temporal-window to adjust window size.

Pre-processing (applied before AI upscaling to clean degraded sources):
  --prefilter LEVEL         none, light (default), medium, heavy
                              none       No filtering — use for clean/high-quality sources
                              light      Mild temporal denoise (default — safe for most content)
                              medium     Denoise + deblock (for visibly compressed/blocky sources)
                              heavy      Strong denoise + deblock + deringing
  --deinterlace             Deinterlace source (for interlaced TV captures etc.)

Output:
  -o, --output FILE         Output file (default: INPUT_upscaled_RES.mkv)
  -q, --quality QUALITY     high (crf 16), medium (crf 20), low (crf 24) [default: high]
  --sharpen                 Apply mild unsharp mask to final output

Performance / quality (single-frame models only):
  -t, --tile SIZE           Tile size for GPU processing (auto by source resolution)
                              0      Full-frame — no tiling (auto for ≤720p, RRDB models only)
                              1024   2×2 tiles (auto for 1080p)
                              512    3×3+ tiles (auto for 1440p/2160p)
                            Note: do NOT use -t 0 with transformer models (nomos8kdat, hat)
                            Reduce on low VRAM: 256, 128
  --tile-pad SIZE           Tile overlap padding in pixels (default: 64)
                            Increase to reduce seam artifacts; decrease to save VRAM
  --full-precision          Use float32 instead of float16 (marginal quality gain, uses more VRAM)

Temporal model options:
  --temporal-window N       Sliding window size in frames (default: 15)
                            Larger = more temporal context, more VRAM. Reduce if OOM.

Workflow:
  --resume                  Resume interrupted run — skip already-completed frames
  --keep-temp               Keep temporary files after completion
  -h, --help                Show this help

Examples:
  # Standard — compressed broadcast or web download
  $0 -i film.mkv -r 1080p

  # Temporal — best for flickery/heavily compressed sources
  $0 -i film.mkv -r 1080p -m basicvsr

  # Temporal with smaller window (reduces VRAM)
  $0 -i film.mkv -r 1080p -m basicvsr --temporal-window 7

  # Heavily degraded source
  $0 -i old_capture.mkv -r 1080p --prefilter heavy --deinterlace

  # Clean source, highest fidelity single-frame model
  $0 -i bluray_rip.mkv -r 2160p -m hat --prefilter none

  # Resume an interrupted run
  $0 -i film.mkv -r 1080p --resume

  # 4K with sharpening, custom tile size
  $0 -i film.mkv -r 2160p -t 256 --sharpen

Model downloads (place .pth files in $MODEL_DIR):
  nomos8k      openmodeldb.info                           → 4xNomos8kSC.pth
  nomos8kdat   openmodeldb.info                           → 4xNomos8kDAT.pth
  lsdir        github.com/Phhofm/models                  → 4xLSDIR.pth
  ultrasharp   huggingface.co/Kim2091/UltraSharp          → 4x-UltraSharp.pth
  realesrgan   github.com/xinntao/Real-ESRGAN             → RealESRGAN_x4plus.pth
  hat          github.com/XPixelGroup/HAT/releases        → HAT-L_SRx4_ImageNet-pretrain.pth
               (requires: pip install spandrel-extra-arches)
  basicvsr     openmmlab CDN                              → BasicVSR_PlusPlus_REDS4.pth
               wget -O ~/ai-upscale/models/BasicVSR_PlusPlus_REDS4.pth \
               https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth
               (requires: pip install basicsr)
  spynet       Required by both temporal models           → spynet_20210409-c6c1bd09.pth
               (downloaded automatically by basicsr on first use)
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
    TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$f" | head -1)
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

    local scan_label="progressive"
    [[ "$IS_INTERLACED" == true ]] && scan_label="interlaced"
    print_info "Input: ${INPUT_WIDTH}x${INPUT_HEIGHT}  SAR:${INPUT_SAR:-1:1}  DAR:${INPUT_DAR:-N/A}  ${INPUT_FPS}fps  ${DURATION}s  ${TOTAL_FRAMES} frames  codec:${INPUT_CODEC}  scan:${scan_label}  audio:${HAS_AUDIO}"
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
    # Extract from original input (cleaned_source.mkv has -an / no audio)
    ffmpeg -i "$input_file" -vn -acodec copy "$audio_file" -y -loglevel error || true

    if [[ -f "$audio_file" && -s "$audio_file" ]]; then
        print_success "Audio extracted"
    else
        HAS_AUDIO=false
        print_warning "Audio extraction failed — encoding without audio"
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

    # Use ffprobe for reliable fps/frame-count (OpenCV is unreliable on MKV/FFV1)
    import subprocess, json as _json
    _probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', 'stream=r_frame_rate,nb_read_frames',
         '-of', 'json', input_video],
        capture_output=True, text=True)
    _info = _json.loads(_probe.stdout)['streams'][0]
    _num, _den = map(int, _info['r_frame_rate'].split('/'))
    fps = _num / _den
    total_frames = int(_info.get('nb_read_frames', 0)) or None

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
                print(f"\n[WARN] Frame {frame_num} failed ({e}) — writing resized original", flush=True)
                try:
                    fallback = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_LANCZOS4)
                    write_q.put((frame_path, fallback))
                except Exception:
                    pass

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
        tile_size          = int(tile_size),
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
import cv2
import torch
import numpy as np
from tqdm import tqdm

# Reduce CUDA memory fragmentation. BasicVSR++ makes many large allocations during
# propagation; expandable segments lets PyTorch reuse fragmented reserved blocks.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

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
    _probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', 'stream=r_frame_rate,nb_read_frames,width,height',
         '-of', 'json', input_video],
        capture_output=True, text=True)
    _info = _json.loads(_probe.stdout)['streams'][0]
    _num, _den = map(int, _info['r_frame_rate'].split('/'))
    fps = _num / _den
    total_frames = int(_info.get('nb_read_frames', 0)) or 0
    src_w = int(_info['width'])
    src_h = int(_info['height'])

    with open(os.path.join(frames_dir, 'fps.txt'), 'w') as fh:
        fh.write(f"{fps:.6f}")

    # Resume: if all expected frames already exist, skip entirely
    if resume:
        existing = [f for f in os.listdir(frames_dir)
                    if f.startswith('frame_') and f.endswith('.png')]
        if len(existing) >= max(total_frames, 1):
            print(f"Resume: {len(existing)} frames already present — skipping upscale")
            return True

    # ── Streaming sliding-window inference ────────────────────────────────────
    # The model processes a window of `window_size` frames at once.
    # We keep OVERLAP frames of past context and read-ahead for future context
    # so boundaries between windows have full neighbour information.
    #
    #  window = [past_overlap | new_frames | future_overlap]
    #  valid output = output[past_overlap : past_overlap + new_frames]
    #
    # Only `window_size` raw frames are in RAM at any moment.
    # ─────────────────────────────────────────────────────────────────────────
    OVERLAP = min(2, window_size // 4)
    stride  = max(1, window_size - 2 * OVERLAP)

    print(f"Temporal window: {window_size}  |  Overlap: {OVERLAP}  |  Stride: {stride}")
    print(f"Output: {output_w}×{output_h}")

    # ── FFmpeg pipe reader (reliable for all containers, unlike OpenCV) ────────
    ffmpeg_cmd = [
        'ffmpeg', '-v', 'error', '-i', input_video,
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE,
                                   bufsize=src_w * src_h * 3 * 4)
    frame_nbytes = src_w * src_h * 3

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

    output_idx = 0
    is_first   = True

    with tqdm(total=total_frames if total_frames > 0 else None,
              desc="Upscaling frames (temporal)", unit="frame") as pbar:
        while True:
            # Build [1, T, C, H, W] batch
            tensors = torch.stack(
                [frame_to_tensor(f, device, use_half) for f in window], dim=0
            ).unsqueeze(0)

            with torch.no_grad():
                out_batch = model(tensors)  # [1, T, C, 4H, 4W]

            # Which frames from this window are "valid" output?
            valid_start = 0 if is_first else OVERLAP
            valid_end   = len(window) if eof else len(window) - OVERLAP

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

            if eof:
                break

            is_first = False

            # Slide: keep 2*OVERLAP trailing frames so the next window is full-size
            # and the trimmed middle region [OVERLAP:-OVERLAP] is contiguous.
            context    = window[-(2 * OVERLAP):] if OVERLAP > 0 else []
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
        window_size        = int(window_size),
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
        print_info "Model: $MODEL_KEY  |  Window: ${TEMPORAL_WINDOW} frames  |  Prefilter: ${PREFILTER}"

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
        [[ "$TILE_SIZE" == "0" ]] && tile_note="0 (full-frame)"
        [[ "$TILE_SIZE_EXPLICIT" == false ]] && tile_note+=" (auto)"
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
    encode_output "$frames_dir" "$output_file"
}

encode_output() {
    local frames_dir="$1"
    local output_file="$2"

    local fps
    fps=$(cat "$frames_dir/fps.txt")
    print_info "Encoding: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT} @ ${fps}fps → $output_file"

    local crf
    case "$QUALITY" in
        high)   crf=16 ;;
        medium) crf=20 ;;
        low)    crf=24 ;;
        *)      print_error "Unknown quality: $QUALITY"; exit 1 ;;
    esac

    # Final resize to exact target dimensions (safety net for rounding differences)
    # plus optional sharpening
    local vf_out="scale=${OUTPUT_WIDTH}:${OUTPUT_HEIGHT}:flags=lanczos"
    if [[ "$SHARPEN" == true ]]; then
        vf_out="${vf_out},unsharp=3:3:0.5:3:3:0.0"
    fi

    if [[ "$HAS_AUDIO" == true && -f "$TEMP_DIR/audio.mka" ]]; then
        ffmpeg \
            -framerate "$fps" \
            -pattern_type glob -i "$frames_dir/frame_*.png" \
            -i "$TEMP_DIR/audio.mka" \
            -vf "$vf_out" \
            -c:v libx265 -crf "$crf" -preset slow \
            -pix_fmt yuv420p10le \
            -x265-params "no-open-gop=1:keyint=250:bframes=8:aq-mode=3" \
            -c:a copy \
            -movflags +faststart \
            "$output_file" -y -loglevel error -stats
    else
        ffmpeg \
            -framerate "$fps" \
            -pattern_type glob -i "$frames_dir/frame_*.png" \
            -vf "$vf_out" \
            -c:v libx265 -crf "$crf" -preset slow \
            -pix_fmt yuv420p10le \
            -x265-params "no-open-gop=1:keyint=250:bframes=8:aq-mode=3" \
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

    ffmpeg -i "$input_file" \
        -vf "scale=${OUTPUT_WIDTH}:${OUTPUT_HEIGHT}:flags=lanczos" \
        -c:v libx265 -crf "$crf" -preset slow \
        -pix_fmt yuv420p10le \
        -c:a copy \
        -movflags +faststart \
        "$output_file" -y -loglevel error -stats

    print_success "Output saved: $output_file"
}

cleanup() {
    if [[ "$KEEP_TEMP" == false ]]; then
        print_info "Cleaning up temp files..."
        rm -rf "$TEMP_DIR"
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
        --prefilter)         PREFILTER="$2";       shift 2 ;;
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
# Main
##############################################################################

if [[ -z "$INPUT_FILE" || -z "$RESOLUTION" ]]; then
    print_error "Missing required arguments (-i and -r)"
    usage
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
    rm -rf "$TEMP_DIR/frames"
    rm -f  "$TEMP_DIR/cleaned_source.mkv"
    rm -f  "$TEMP_DIR/upscale.py"
    rm -f  "$TEMP_DIR/upscale_temporal.py"
    rm -f  "$TEMP_DIR/audio.mka"
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

    # Auto-select tile size for single-frame models (not used by temporal pipeline).
    # Full-frame (tile=0) is only beneficial for RRDB-based models (nomos8k, realesrgan,
    # lsdir, ultrasharp) — transformer models (nomos8kdat, hat, span) run dramatically
    # slower at tile=0 due to attention mechanisms. Use -t 0 explicitly only when you
    # know you are running an RRDB model and have enough VRAM.
    if [[ "$IS_TEMPORAL" == false && "$TILE_SIZE_EXPLICIT" == false ]]; then
        if (( INPUT_HEIGHT <= 720 )); then
            TILE_SIZE=0        # full-frame — fastest for RRDB models on ≤720p with 16GB VRAM
        elif (( INPUT_HEIGHT <= 1080 )); then
            TILE_SIZE=1024     # 2×2 tiles for 1080p
        fi
        # 1440p / 2160p keep the 512 default
    fi

    extract_audio "$INPUT_FILE"
    run_prefilter "$INPUT_FILE"
    upscale_video "$OUTPUT_FILE"
else
    simple_scale "$INPUT_FILE" "$OUTPUT_FILE"
fi

cleanup

print_success "=== Complete ==="
OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" 2>/dev/null | cut -f1 || echo "unknown")
print_success "Output: $OUTPUT_FILE ($OUTPUT_SIZE)"
