#!/bin/bash

##############################################################################
# AI Video Upscaler — spandrel edition
# Models: nomos8k (default), lsdir, ultrasharp, realesrgan, hat
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
TILE_PAD=32
MODEL_KEY="nomos8k"
QUALITY="high"
PREFILTER="medium"
DEINTERLACE=false
RESUME=false
SHARPEN=false
FULL_PRECISION=false
KEEP_TEMP=false
UPSCALE_SOURCE=""

# ── Model registry (all 4x; outscale handles sub-4x targets) ──────────────────
declare -A MODEL_FILES=(
    [nomos8k]="4xNomos8kSC.pth"
    [lsdir]="4xLSDIR.pth"
    [ultrasharp]="4x-UltraSharp.pth"
    [realesrgan]="RealESRGAN_x4plus.pth"
    [hat]="HAT-L_SRx4_ImageNet-pretrain.pth"
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
${GREEN}AI Video Upscaler${NC} — spandrel edition, live-action optimised

Usage: $0 -i INPUT -r RESOLUTION [OPTIONS]

Required:
  -i, --input FILE          Input video file
  -r, --resolution RES      Target resolution: 720p, 1080p, 1440p, 2160p

Model selection:
  -m, --model TYPE          Upscale model (default: nomos8k)
                              nomos8k    Best all-round for compressed live-action  ← default
                              lsdir      Sharp detail, handles real-world degradations
                              ultrasharp Maximum sharpness (better on cleaner sources)
                              realesrgan Real-ESRGAN x4plus (legacy fallback)
                              hat        HAT-L transformer — highest fidelity, clean sources only

Pre-processing (applied before AI upscaling to clean degraded sources):
  --prefilter LEVEL         none, light, medium (default), heavy
                              none       No filtering — raw input to upscaler
                              light      Mild temporal denoise
                              medium     Denoise + deblock (recommended for compressed sources)
                              heavy      Strong denoise + deblock + deringing
  --deinterlace             Deinterlace source (for interlaced TV captures etc.)

Output:
  -o, --output FILE         Output file (default: INPUT_upscaled_RES.mkv)
  -q, --quality QUALITY     high (crf 16), medium (crf 20), low (crf 24) [default: high]
  --sharpen                 Apply mild unsharp mask to final output

Performance / quality:
  -t, --tile SIZE           Tile size for GPU processing (default: 512)
                            Reduce on low VRAM: 256, 128
  --tile-pad SIZE           Tile overlap padding in pixels (default: 32)
                            Increase to reduce seam artifacts
  --full-precision          Use float32 instead of float16 (marginal quality gain, uses more VRAM)

Workflow:
  --resume                  Resume interrupted run — skip already-completed frames
  --keep-temp               Keep temporary files after completion
  -h, --help                Show this help

Examples:
  # Standard — compressed broadcast or web download
  $0 -i film.mkv -r 1080p

  # Heavily degraded source
  $0 -i old_capture.mkv -r 1080p --prefilter heavy --deinterlace

  # Clean source, highest fidelity model
  $0 -i bluray_rip.mkv -r 2160p -m hat --prefilter light

  # Resume an interrupted run
  $0 -i film.mkv -r 1080p --resume

  # 4K with sharpening, custom tile size
  $0 -i film.mkv -r 2160p -t 256 --sharpen

Model downloads (place .pth files in $MODEL_DIR):
  nomos8k    https://github.com/Phhofm/models              → 4xNomos8kSC.pth
  lsdir      https://github.com/Phhofm/models              → 4xLSDIR.pth
  ultrasharp https://huggingface.co/Kim2091/UltraSharp     → 4x-UltraSharp.pth
  realesrgan https://github.com/xinntao/Real-ESRGAN        → RealESRGAN_x4plus.pth
  hat        https://github.com/XPixelGroup/HAT/releases   → HAT-L_SRx4_ImageNet-pretrain.pth
             (requires: pip install spandrel-extra-arches)
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

    INPUT_WIDTH=$(ffprobe  -v error -select_streams v:0 -show_entries stream=width            -of csv=p=0 "$f")
    INPUT_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height           -of csv=p=0 "$f")
    INPUT_FPS=$(ffprobe    -v error -select_streams v:0 -show_entries stream=r_frame_rate     -of csv=p=0 "$f" | bc -l | xargs printf "%.3f")
    DURATION=$(ffprobe     -v error                     -show_entries format=duration         -of csv=p=0 "$f" | xargs printf "%.2f")
    TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$f")
    INPUT_CODEC=$(ffprobe  -v error -select_streams v:0 -show_entries stream=codec_name       -of csv=p=0 "$f")

    local audio_stream
    audio_stream=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of csv=p=0 "$f" 2>/dev/null || true)
    HAS_AUDIO=$([[ -n "$audio_stream" ]] && echo true || echo false)

    print_info "Input: ${INPUT_WIDTH}x${INPUT_HEIGHT}  ${INPUT_FPS}fps  ${DURATION}s  ${TOTAL_FRAMES} frames  codec:${INPUT_CODEC}  audio:${HAS_AUDIO}"
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

    OUTPUT_HEIGHT=$TARGET_HEIGHT
    OUTPUT_WIDTH=$(echo "$INPUT_WIDTH * $TARGET_HEIGHT / $INPUT_HEIGHT" | bc | cut -d. -f1)
    OUTPUT_WIDTH=$(( (OUTPUT_WIDTH / 2) * 2 ))
    OUTPUT_HEIGHT=$(( (OUTPUT_HEIGHT / 2) * 2 ))

    print_info "Scale factor: ${SCALE_FACTOR}x  →  Output: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
}

select_model() {
    if [[ -z "${MODEL_FILES[$MODEL_KEY]+_}" ]]; then
        print_error "Unknown model: $MODEL_KEY"
        print_error "Available: ${!MODEL_FILES[*]}"
        exit 1
    fi

    MODEL_PATH="$MODEL_DIR/${MODEL_FILES[$MODEL_KEY]}"

    if [[ ! -f "$MODEL_PATH" ]]; then
        print_error "Model not found: $MODEL_PATH"
        print_error "Download it and place in: $MODEL_DIR  (see --help for URLs)"
        exit 1
    fi

    print_info "Model: $MODEL_KEY  (${MODEL_FILES[$MODEL_KEY]})"
}

run_prefilter() {
    local source_file="$1"
    local cleaned="$TEMP_DIR/cleaned_source.mkv"
    UPSCALE_SOURCE="$source_file"

    local vf_parts=()

    if [[ "$DEINTERLACE" == true ]]; then
        # mode=1: output a frame for each field (doubles framerate for interlaced content)
        vf_parts+=("yadif=mode=1:parity=-1:deint=1")
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

# Transformer models (HAT, SwinIR) require input dimensions to be a multiple of
# their window size. 64 is a safe LCM for all supported architectures.
WINDOW_MULTIPLE = 64


def pad_to_multiple(tensor, multiple):
    """Pad H and W dims of an NCHW tensor to the nearest multiple. Returns (padded, pad_w, pad_h)."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, 0, 0
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded, pad_w, pad_h


def frame_to_tensor(frame, device, use_half):
    """Convert cv2 BGR uint8 HWC frame to NCHW float [0,1] tensor on device."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    if use_half:
        t = t.half()
    return t


def tensor_to_frame(t):
    """Convert NCHW float [0,1] tensor to cv2 BGR uint8 HWC frame."""
    out = t.squeeze(0).float().cpu().clamp(0, 1)
    out = (out.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def tile_process(model, img_tensor, model_scale, tile_size, tile_pad):
    """
    Tile-based inference with per-tile WINDOW_MULTIPLE alignment padding.

    Tiles are padded with tile_pad pixels of context on each side for better
    edge quality, then further padded to WINDOW_MULTIPLE for transformer model
    compatibility (harmless no-op for RRDBNet-based models).
    """
    batch, channel, height, width = img_tensor.shape
    # Accumulate in float32 regardless of inference dtype
    output = torch.zeros(batch, channel, height * model_scale, width * model_scale,
                         dtype=torch.float32)

    tiles_x = math.ceil(width  / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            # Core tile bounds (no padding)
            x0 = tile_x * tile_size
            x1 = min(x0 + tile_size, width)
            y0 = tile_y * tile_size
            y1 = min(y0 + tile_size, height)

            # Expanded bounds with tile_pad context for better edge quality
            x0p = max(x0 - tile_pad, 0)
            x1p = min(x1 + tile_pad, width)
            y0p = max(y0 - tile_pad, 0)
            y1p = min(y1 + tile_pad, height)

            tile_in = img_tensor[:, :, y0p:y1p, x0p:x1p]

            # Pad to WINDOW_MULTIPLE for transformer model compatibility
            tile_in, pw, ph = pad_to_multiple(tile_in, WINDOW_MULTIPLE)

            with torch.no_grad():
                tile_out = model(tile_in).float()

            # Strip WINDOW_MULTIPLE padding from output
            h_content = tile_out.shape[2] - ph * model_scale
            w_content = tile_out.shape[3] - pw * model_scale
            tile_out = tile_out[:, :, :h_content, :w_content]

            # Extract only the core (non-tile_pad) region and place in output
            off_x   = (x0 - x0p) * model_scale
            off_y   = (y0 - y0p) * model_scale
            w_tile  = (x1 - x0)  * model_scale
            h_tile  = (y1 - y0)  * model_scale

            output[
                :, :,
                y0 * model_scale : y1 * model_scale,
                x0 * model_scale : x1 * model_scale,
            ] = tile_out[:, :, off_y : off_y + h_tile, off_x : off_x + w_tile]

    return output


def upscale_video(input_video, frames_dir, model_path,
                  tile_size, tile_pad, outscale, use_full_precision, resume):

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

    print(f"Model native scale: {model_scale}x  |  Target outscale: {outscale:.4f}x")

    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None  # unknown — tqdm will show count only

    with open(os.path.join(frames_dir, 'fps.txt'), 'w') as f:
        f.write(str(fps))

    frame_num = 0
    skipped   = 0

    with tqdm(total=total_frames, desc="Upscaling frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(frames_dir, f'frame_{frame_num:08d}.png')

            # Resume: skip frames already on disk
            if resume and os.path.isfile(frame_path):
                frame_num += 1
                skipped   += 1
                pbar.update(1)
                continue

            try:
                h, w    = frame.shape[:2]
                img_t   = frame_to_tensor(frame, device, use_half)

                if tile_size > 0:
                    out_t = tile_process(model, img_t, model_scale, tile_size, tile_pad)
                else:
                    # Full-image inference (only viable for small frames with large VRAM)
                    img_padded, pw, ph = pad_to_multiple(img_t, WINDOW_MULTIPLE)
                    with torch.no_grad():
                        out_t = model(img_padded).float()
                    if ph > 0 or pw > 0:
                        out_t = out_t[
                            :, :,
                            : out_t.shape[2] - ph * model_scale,
                            : out_t.shape[3] - pw * model_scale,
                        ]

                out_frame = tensor_to_frame(out_t)

                # Resize from model's native 4x to target outscale
                # INTER_AREA for downscaling (e.g. 4x → 2.25x), INTER_LANCZOS4 otherwise
                target_h = round(h * outscale)
                target_w = round(w * outscale)
                if out_frame.shape[0] != target_h or out_frame.shape[1] != target_w:
                    interp = cv2.INTER_AREA if outscale < model_scale else cv2.INTER_LANCZOS4
                    out_frame = cv2.resize(out_frame, (target_w, target_h), interpolation=interp)

                # PNG compression=1: fastest write, larger files — correct for temp frames
                cv2.imwrite(frame_path, out_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            except Exception as e:
                print(f"\n[WARN] Frame {frame_num} failed: {e}", flush=True)

            frame_num += 1
            pbar.update(1)

    cap.release()

    if skipped > 0:
        print(f"Resumed: skipped {skipped} already-completed frames")

    return frame_num > 0


if __name__ == '__main__':
    _, input_video, frames_dir, model_path, \
        tile_size, tile_pad, outscale, \
        use_full_precision_str, resume_str = sys.argv

    success = upscale_video(
        input_video        = input_video,
        frames_dir         = frames_dir,
        model_path         = model_path,
        tile_size          = int(tile_size),
        tile_pad           = int(tile_pad),
        outscale           = float(outscale),
        use_full_precision = (use_full_precision_str == 'true'),
        resume             = (resume_str == 'true'),
    )
    sys.exit(0 if success else 1)
PYTHON_EOF
}

upscale_video() {
    local output_file="$1"
    local frames_dir="$TEMP_DIR/frames"

    mkdir -p "$frames_dir"
    source "$VENV_DIR/bin/activate"
    write_python_script

    print_info "Starting AI upscaling..."
    print_info "Model: $MODEL_KEY  |  Tile: ${TILE_SIZE}  |  Tile-pad: ${TILE_PAD}  |  Prefilter: ${PREFILTER}"

    "$VENV_DIR/bin/python3" "$TEMP_DIR/upscale.py" \
        "$UPSCALE_SOURCE" \
        "$frames_dir" \
        "$MODEL_PATH" \
        "$TILE_SIZE" \
        "$TILE_PAD" \
        "$SCALE_FACTOR" \
        "$FULL_PRECISION" \
        "$RESUME"

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
        -i|--input)       INPUT_FILE="$2";      shift 2 ;;
        -o|--output)      OUTPUT_FILE="$2";     shift 2 ;;
        -r|--resolution)  RESOLUTION="$2";      shift 2 ;;
        -m|--model)       MODEL_KEY="$2";       shift 2 ;;
        -t|--tile)        TILE_SIZE="$2";       shift 2 ;;
        --tile-pad)       TILE_PAD="$2";        shift 2 ;;
        -q|--quality)     QUALITY="$2";         shift 2 ;;
        --prefilter)      PREFILTER="$2";       shift 2 ;;
        --deinterlace)    DEINTERLACE=true;     shift   ;;
        --resume)         RESUME=true;          shift   ;;
        --sharpen)        SHARPEN=true;         shift   ;;
        --full-precision) FULL_PRECISION=true;  shift   ;;
        --keep-temp)      KEEP_TEMP=true;       shift   ;;
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
