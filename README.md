# AI Video Upscaler

AI-powered video upscaling for Ubuntu with NVIDIA GPU acceleration. Built around the spandrel and basicsr model loaders with a focus on live-action content — low resolution, compressed, noisy, and artifact-heavy sources.

## System Requirements

- OS: Ubuntu 24.04 LTS
- GPU: NVIDIA GPU with CUDA support (8GB+ VRAM recommended, 16GB for temporal models and 4K)
- RAM: 2GB minimum, 4GB+ recommended
- Storage: 20GB+ free for models and temp files (500GB+ for 24fps 2-hour video temp frames)
- Internet: Required for initial setup

## Installation

```bash
git clone https://github.com/gitorion/ai-upscaler.git
cd ai-upscaler
chmod +x install.sh
./install.sh

# After reboot (if driver install prompted one):
./install.sh --resume
```

The installer handles NVIDIA drivers, CUDA, FFmpeg, Python environment, core dependencies, and model downloads (BasicVSR++ and SPyNet).

## Model Setup

The installer downloads BasicVSR++ and SPyNet weights automatically. The default single-frame model (`4xNomos8kSC.pth`) must be downloaded manually to `~/ai-upscale/models/`.

### Single-frame models (spandrel)

Process one frame at a time. No additional dependencies beyond the base install.

| Key | Filename | Best for | Speed (4060 Ti 16GB) | Download |
|-----|----------|----------|----------------------|----------|
| `nomos8k` | `4xNomos8kSC.pth` | Compressed live-action — **default** | ~4 s/frame @ 480p | openmodeldb.info |
| `ultrasharp` | `4x-UltraSharp.pth` | Maximum sharpness on clean sources | ~4 s/frame @ 480p | huggingface.co/Kim2091/UltraSharp |
| `lsdir` | `4xLSDIR.pth` | Sharp detail, real-world degradations | ~4 s/frame @ 480p | openmodeldb.info |
| `hat` | `HAT-L_SRx4_ImageNet-pretrain.pth` | Highest fidelity transformer, clean sources only | ~8 s/frame @ 480p | github.com/XPixelGroup/HAT/releases |
| `nomos8kdat` | `4xNomos8kDAT.pth` | DAT transformer — very slow, short clips only | ~24 s/frame @ 480p | openmodeldb.info |
| `realesrgan` | `RealESRGAN_x4plus.pth` | Legacy fallback | ~4 s/frame @ 480p | github.com/xinntao/Real-ESRGAN/releases |

Note: `hat` additionally requires `pip install spandrel-extra-arches` in the venv.

### Temporal models (basicsr)

Process a sliding window of frames simultaneously using optical flow. Better temporal consistency and dramatically faster than single-frame models — the only practical option for full-length TV episodes and movies.

| Key | Filename | Best for | Speed (4060 Ti 16GB) | Download |
|-----|----------|----------|----------------------|----------|
| `basicvsr` | `BasicVSR_PlusPlus_REDS4.pth` | Long content, degraded/compressed video | ~0.16 s/frame @ 480p, ~2 s/frame @ 720p | openmmlab CDN |

Temporal models require:
1. `pip install basicsr` in the venv (handled by `install.sh`)
2. SPyNet optical flow weights (downloaded by `install.sh`):
   ```bash
   wget -O ~/ai-upscale/models/spynet_20210409-c6c1bd09.pth \
     https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth
   ```
3. BasicVSR++ weights (downloaded by `install.sh`):
   ```bash
   wget -O ~/ai-upscale/models/BasicVSR_PlusPlus_REDS4.pth \
     https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth
   ```

All models are 4x. The script uses the model's native 4x output and resizes to your target resolution at the FFmpeg encode step.

### Estimated total upscale time (1 hour, 25fps, 4060 Ti 16GB)

| Model | 480p source | 720p source |
|-------|------------|------------|
| basicvsr | ~4 hrs | ~51 hrs |
| nomos8k / ultrasharp / lsdir | ~100 hrs | ~250 hrs |
| hat | ~200 hrs | ~500 hrs |
| nomos8kdat | ~600 hrs | ~1600 hrs |

BasicVSR++ is 5-15x faster than single-frame models and is the recommended choice for any content longer than a few minutes.

## Quick Start

### Interactive mode

Run with just an input file and the script walks you through every option:

```bash
./upscale_video.sh -i input.mkv
```

You'll be prompted for resolution, model, prefilter level, and deinterlace — each with descriptions to guide your choice.

### CLI mode

Pass all options directly for scripted or automated use:

```bash
# Standard — compressed broadcast rip (nomos8k default)
./upscale_video.sh -i recording.mkv -r 1080p

# Temporal — best for long content and degraded sources
./upscale_video.sh -i recording.mkv -r 1080p -m basicvsr

# Heavily degraded source — maximum pre-processing + deinterlace
./upscale_video.sh -i old_vhs.mkv -r 1080p --prefilter heavy --deinterlace -m basicvsr

# Clean source — highest quality single-frame model, no pre-processing
./upscale_video.sh -i bluray_rip.mkv -r 2160p -m hat --prefilter none

# Temporal with smaller window to save VRAM
./upscale_video.sh -i recording.mkv -r 1080p -m basicvsr --temporal-window 7

# Faster encode — trade some compression efficiency for ~4x faster encode
./upscale_video.sh -i recording.mkv -r 1080p -m basicvsr --encode-speed fast

# Resume an interrupted run
./upscale_video.sh -i recording.mkv -r 1080p --resume

# Batch process a folder
for f in /path/to/videos/*.mkv; do
    ./upscale_video.sh -i "$f" -r 1080p -m basicvsr
done
```

## All Options

```
Usage: upscale_video.sh -i INPUT [-r RESOLUTION] [OPTIONS]

  Run with just -i for interactive mode, or pass all options via CLI.

Required:
  -i, --input FILE          Input video file
  -r, --resolution RES      Target: 720p, 1080p, 1440p, 2160p
                             (omit for interactive mode)

Model:
  -m, --model TYPE          nomos8k (default), ultrasharp, lsdir, hat,
                            nomos8kdat, realesrgan, basicvsr

Pre-processing:
  --prefilter LEVEL         none, light (default), medium, heavy
  --deinterlace             Deinterlace source before upscaling
                            (auto-detects interlaced vs progressive)

Output:
  -o, --output FILE         Output filename (default: INPUT_upscaled_RES.mkv)
  -q, --quality LEVEL       high (default, crf 16), medium (crf 20), low (crf 24)
  --encode-speed SPEED      slow (default, best quality), medium (~2x faster), fast (~4x faster)
  --sharpen                 Apply unsharp mask to final output

Performance / quality (single-frame models):
  -t, --tile SIZE           Tile size (default: auto — probes GPU VRAM after model load)
  --tile-pad SIZE           Tile overlap padding (default: 64)
  --full-precision          Use float32 instead of float16 (auto for hat/nomos8kdat)

Temporal model options:
  --temporal-window N       Sliding window size in frames (default: auto — probes GPU)

Workflow:
  --resume                  Skip already-completed frames from an interrupted run
  --keep-temp               Keep temp files after completion
  -h, --help                Show full help
```

## Flag Reference

### --prefilter

Runs an FFmpeg pass over the source before the AI sees it, using lossless FFV1 encoding as the intermediate. Cleaning compression artifacts before upscaling produces sharper, more accurate results.

| Level | What it applies | Best for |
|-------|----------------|----------|
| `none` | Raw input | Clean sources: Blu-ray rips, high-quality 1080p |
| `light` | Mild temporal denoise | **Default** — safe for most content |
| `medium` | Denoise + deblock | Compressed/blocky sources: web downloads, streaming rips |
| `heavy` | Strong denoise + deblock + deringing | Badly degraded: VHS, old TV recordings, heavy compression |

### --deinterlace

Applies a yadif deinterlace pass before the prefilter. The script auto-detects whether the source is interlaced or progressive via ffprobe field order analysis:

- **Interlaced source**: uses `yadif mode=1` (one frame per field — doubles framerate to preserve temporal detail)
- **Progressive source**: uses `yadif mode=0` (framerate preserved — safe no-op)

Use for: old TV recordings, broadcast captures, DVD rips, VHS sources. Safe to enable on progressive sources — it won't double the framerate unnecessarily.

### --model

All models are 4x. Selection guide:

**Single-frame (spandrel) — process one frame at a time:**

- **nomos8k** — RRDB-based, trained on real-world degradations. Fast and reliable. Best all-rounder for compressed video. **Default.**
- **ultrasharp** — Maximum perceived sharpness on clean sources. Can over-sharpen noisy inputs.
- **lsdir** — Sharp edges and fine detail on real-world degraded content.
- **hat** — Hybrid Attention Transformer. Highest fidelity single-frame model, designed for clean sources. Pair with `--prefilter none`. Slow (~2x nomos8k).
- **nomos8kdat** — DAT transformer. Very slow (~6x nomos8k) — short clips only.
- **realesrgan** — The original RealESRGAN x4plus. Kept as a legacy fallback.

**Temporal (basicsr) — process a sliding window of frames:**

- **basicvsr** — BasicVSR++ with bidirectional propagation and optical flow alignment. 5-15x faster than single-frame models. Better temporal consistency and less flickering. The only practical model for full-length TV episodes and movies on consumer hardware. Best for degraded/compressed sources where noise is consistent across frames.

### --temporal-window

Controls the sliding window size for temporal models. Defaults to **auto** — after loading the model, the script probes GPU VRAM by running a test window with descending sizes (15 → 13 → 11 → 9 → 7 → 5 → 3) and selects the largest that fits. A larger window gives better temporal continuity for slow pans and smooth motion. Pass `--temporal-window N` to override.

### --tile / --tile-pad

Tile size defaults to **auto** — after loading the model, the script probes GPU VRAM by running a test frame with descending tile sizes (full-frame → 768 → 512 → 384 → 256 → 192 → 128) and selects the largest size that fits. This automatically adapts to any GPU, model, and precision combination.

Pass `-t SIZE` to override with a specific value. Tile-pad controls how many pixels of overlap context each tile borrows from its neighbours — the default of 64 is a good balance. There is no quality loss from tiling.

### --resume

The upscaler writes each frame as a PNG to a temp folder as it goes. If a run is interrupted, rerun the exact same command with `--resume` added and it will skip any frames already on disk, picking up where it left off.

### --full-precision

Uses float32 instead of float16 for model inference. Auto-enabled for transformer models (`hat`, `nomos8kdat`) which overflow float16. For RRDB models the quality difference is marginal. Uses roughly 2x the VRAM — the auto tile probe accounts for this.

### --sharpen

Adds `unsharp=3:3:0.5:3:3:0.0` to the FFmpeg encode step. Gives a crisper look on soft or lightly upscaled content. Avoid on already-sharp sources.

### --quality

Controls the HEVC output encode quality (CRF). Output is always HEVC 10-bit in an MKV container.

| Level | CRF | Use case |
|-------|-----|----------|
| `high` | 16 | **Default** — visually lossless |
| `medium` | 20 | Good balance of quality and file size |
| `low` | 24 | Smaller files, some visible compression |

### --encode-speed

Controls the x265 encoder preset. This affects the final encode step only — it has no impact on AI upscaling speed.

| Speed | x265 preset | Encode time (relative) | Notes |
|-------|-------------|----------------------|-------|
| `slow` | slow | 1x (baseline) | **Default** — best compression efficiency and quality |
| `medium` | medium | ~2x faster | Good balance for long content |
| `fast` | fast | ~4x faster | Noticeably faster, slightly larger files at same CRF |

For a 97-minute 1080p 60fps encode, `slow` takes ~9 hours on an 8-thread CPU. Switch to `medium` or `fast` when encode time matters more than squeezing out maximum quality per byte.

## Pipeline

The upscaler runs a multi-stage pipeline:

1. **Analysis** — ffprobe reads source resolution, framerate, SAR/DAR, field order, codec
2. **Audio extraction** — audio stream copied losslessly for remuxing
3. **Pre-filter** — optional deinterlace + denoise/deblock via FFmpeg (lossless FFV1 intermediate)
4. **AI upscaling** — model processes frames via GPU (single-frame tiling or temporal sliding window)
5. **Encode** — upscaled PNGs encoded to HEVC 10-bit with audio remuxed

For temporal models, frame reading uses an ffmpeg pipe (raw BGR24) for reliable decoding of all container formats. GPU memory is managed with `expandable_segments` and explicit cache clearing between windows.

## Aspect Ratio Handling

The script reads SAR (Sample Aspect Ratio) and DAR (Display Aspect Ratio) from the source via ffprobe. For anamorphic sources (e.g. SD DVD: 720x480 stored, SAR 8:9, DAR 4:3), the output width is calculated from the DAR to preserve correct display proportions. Square-pixel sources (SAR 1:1) are unaffected.

## Monitoring a Run

```bash
# Terminal 1: run the upscale
./upscale_video.sh -i film.mkv -r 1080p -m basicvsr

# Terminal 2: watch GPU utilisation
watch -n 1 nvidia-smi
# or for a richer view:
nvtop

# Terminal 3: watch disk space
watch -n 5 df -h
```

### Disk usage

The temp folder (`~/ai-upscale/temp/`) holds upscaled PNG frames at the model's 4x output resolution. Frame size scales with output pixels.

| Output resolution | ~MB per frame | 2-hour film (24fps) |
|-------------------|--------------|---------------------|
| 1080p             | ~3 MB        | ~500 GB             |
| 1440p             | ~6 MB        | ~1 TB               |
| 2160p (4K)        | ~12 MB       | **~2 TB**           |

> **Warning:** A 2-hour 1080p to 2160p upscale needs approximately 2TB of free space for temp frames. Check with `df -h` before starting. If a run runs out of disk space, free space and resume with `--resume`.

## Testing the Installation

```bash
cd ~/ai-upscale
./test.sh
```

## Directory Structure

```
~/ai-upscaler/                  # Git clone (source)
├── upscale_video.sh            # Main upscaling script
├── install.sh                  # Installation script
└── README.md

~/ai-upscale/                   # Runtime directory (created by install.sh)
├── test.sh                     # Installation test script
├── venv/                       # Python virtual environment
├── models/                     # AI model files (.pth)
│   ├── 4xNomos8kSC.pth                  ← default model (nomos8k)
│   ├── BasicVSR_PlusPlus_REDS4.pth      ← basicvsr (temporal)
│   └── spynet_20210409-c6c1bd09.pth     ← required by temporal models
└── temp/                       # Temporary processing files (auto-created, auto-deleted)
```

## Troubleshooting

### GPU not detected
```bash
nvidia-smi
# If this fails:
sudo ubuntu-drivers autoinstall
sudo reboot
```

### CUDA not available in Python
```bash
source ~/ai-upscale/venv/bin/activate
python3 -c "import torch; print(torch.cuda.is_available())"
# If False: reinstall PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### spandrel not found
```bash
source ~/ai-upscale/venv/bin/activate
pip install spandrel spandrel-extra-arches
```

### basicsr not found (temporal models)
```bash
source ~/ai-upscale/venv/bin/activate
pip install basicsr
```

### Out of VRAM
```bash
# Reduce tile size (single-frame models)
./upscale_video.sh -i video.mkv -r 2160p -t 256

# Reduce temporal window (temporal models)
./upscale_video.sh -i video.mkv -r 1080p -m basicvsr --temporal-window 7
```

### PNG inflate errors during encode
A small number of corrupt PNGs (typically < 5 per run) can occur from disk I/O contention during long runs. FFmpeg skips these frames automatically. Not a concern unless the count is high — in that case, check disk health and I/O load.

### libpng write errors during a run
Almost always caused by disk full. Check with `df -h`. Free space and use `--resume` to continue.

### GPU performance
```bash
sudo nvidia-smi -pm 1       # Persistence mode
sudo nvidia-smi -pl 165     # Set power limit (adjust to your GPU's TDP)
```

## Dependencies

- FFmpeg (system)
- Python 3.12 + venv
- PyTorch with CUDA
- spandrel + spandrel-extra-arches (single-frame models)
- basicsr 1.4.2 (temporal models)
- opencv-python, numpy, tqdm

## Credits

- spandrel: github.com/chaiNNer-org/spandrel
- Real-ESRGAN: github.com/xinntao/Real-ESRGAN
- HAT: github.com/XPixelGroup/HAT
- BasicVSR++: github.com/ckkelvinchan/BasicVSR_PlusPlus
- basicsr: github.com/XPixelGroup/BasicSR
- 4xNomos8kSC / 4xLSDIR: github.com/Phhofm/models
- 4x-UltraSharp: huggingface.co/Kim2091
- FFmpeg: ffmpeg.org
- PyTorch: pytorch.org
