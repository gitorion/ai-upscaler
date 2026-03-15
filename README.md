# AI Video Upscaler

AI-powered video upscaling for Ubuntu with NVIDIA GPU acceleration. Built around the spandrel model loader with a focus on live-action content — low resolution, compressed, noisy, and artifact-heavy sources.

## System Requirements

- OS: Ubuntu 24.04 LTS
- GPU: NVIDIA GPU with CUDA support (8GB+ VRAM recommended, 16GB for 4K)
- RAM: 2GB minimum, 4GB+ recommended
- Storage: 20GB+ free for models and temp files (500GB+ for 24fps 2hour video temp frames)
- Internet: Required for initial setup

## Installation

```bash
chmod +x install.sh
./install.sh

# After reboot (if driver install prompted one):
./install.sh --resume
```

The installer handles NVIDIA drivers, CUDA, FFmpeg, Python environment, and core dependencies.

## Model Setup

Models are not included and must be downloaded manually to `~/ai-upscale/models/`.

| Key | Filename | Best for | Download |
|-----|----------|----------|----------|
| `nomos8k` | `4xNomos8kSC.pth` | Compressed live-action — **recommended default** | openmodeldb.info |
| `nomos8kdat` | `4xNomos8kDAT.pth` | Higher quality DAT transformer — ~6x slower, best for short clips | openmodeldb.info |
| `lsdir` | `4xLSDIR.pth` | Sharp detail, real-world degradations | openmodeldb.info |
| `ultrasharp` | `4x-UltraSharp.pth` | Maximum sharpness on cleaner sources | huggingface.co/Kim2091/UltraSharp |
| `realesrgan` | `RealESRGAN_x4plus.pth` | Legacy fallback | github.com/xinntao/Real-ESRGAN/releases |
| `hat` | `HAT-L_SRx4_ImageNet-pretrain.pth` | Highest fidelity, clean sources only | github.com/XPixelGroup/HAT/releases |

All models are 4x. The script uses the model's native 4x output and resizes to your target resolution at the FFmpeg encode step.

`nomos8kdat` uses the DAT (Dual Aggregation Transformer) architecture and produces sharper results with fewer hallucinated artefacts than `nomos8k`. However on current mid-range hardware (e.g. RTX 4060 Ti) it runs ~6x slower (~23s/frame vs ~4s/frame for 720p content), making it impractical for full episode or film upscaling. Reserve it for short clips or stills where quality is the priority.

Note: `hat` additionally requires `pip install spandrel-extra-arches` in the venv.

## Basic Usage

```bash
cd ~/ai-upscale

# Upscale to 1080p (uses nomos8k + light prefilter by default)
./upscale_video.sh -i input.mkv -r 1080p

# Upscale to 4K
./upscale_video.sh -i input.mkv -r 2160p
```

## All Options

```
Required:
  -i, --input FILE          Input video file
  -r, --resolution RES      Target: 720p, 1080p, 1440p, 2160p

Model:
  -m, --model TYPE          nomos8k (default), nomos8kdat, lsdir, ultrasharp, realesrgan, hat

Pre-processing:
  --prefilter LEVEL         none, light (default), medium, heavy
  --deinterlace             Deinterlace source before upscaling

Output:
  -o, --output FILE         Output filename (default: INPUT_upscaled_RES.mkv)
  -q, --quality LEVEL       high (default, crf 16), medium (crf 20), low (crf 24)
  --sharpen                 Apply unsharp mask to final output

Performance / quality:
  -t, --tile SIZE           Tile size (auto-selected by source resolution — override if needed)
                              512   auto for ≤720p and 1440p/2160p
                              1024  auto for 1080p (2×2 tiles)
                              0     Full-frame — RRDB models only (nomos8k, realesrgan,
                                    lsdir, ultrasharp); do NOT use with nomos8kdat or hat
                            Reduce on low VRAM: 256, 128
  --tile-pad SIZE           Tile overlap padding (default: 64)
  --full-precision          Use float32 instead of float16

Workflow:
  --resume                  Skip already-completed frames from an interrupted run
  --keep-temp               Keep temp files after completion
  -h, --help                Show full help
```

## Flag Reference

### --prefilter

Runs an FFmpeg pass over the source before the AI sees it, using lossless ffv1 encoding as the intermediate. Cleaning compression artifacts before upscaling produces sharper, more accurate results.

| Level | What it applies |
|-------|----------------|
| `none` | Raw input — best for clean sources (Blu-ray rips, quality 1080p) |
| `light` | Mild temporal denoise — **default**, safe for most content |
| `medium` | Denoise + deblock — for visibly compressed or blocky sources |
| `heavy` | Strong denoise + deblock + deringing — for badly degraded sources |

### --deinterlace

Prepends a yadif deinterlace pass before the prefilter. Use for interlaced sources — old TV recordings, broadcast captures, VHS rips. Note: yadif mode=1 outputs a frame per field, which doubles the frame count and framerate of the source.

### --model

All models are 4x. Selection guide:

- **nomos8k** — RRDB-based, trained on real-world degradations. Fast (~4s/frame on RTX 4060 Ti for 720p) and reliable. Best choice for batch processing full episodes or films. **Default.**
- **nomos8kdat** — DAT transformer, same training data as nomos8k but higher quality. ~6x slower — practical for short clips or single scenes, not full episodes.
- **lsdir** — tends to produce sharper edges and finer detail on detailed scenes.
- **ultrasharp** — maximum perceived sharpness. Can over-sharpen on already-noisy sources.
- **realesrgan** — the original RealESRGAN x4plus. Kept as a legacy fallback.
- **hat** — Hybrid Attention Transformer. Highest fidelity model but designed for clean (lightly degraded) sources. Pair with `--prefilter none`.

### --tile / --tile-pad

Tile size is auto-selected based on the source resolution:

| Source | Auto tile | Behaviour |
|--------|-----------|-----------|
| ≤ 720p | `0` | Full-frame — no tiling (fastest for RRDB models with 16GB VRAM) |
| 1080p  | `1024` | 2×2 tiles |
| 1440p+ | `512` | 3×3 or more tiles |

Pass `-t SIZE` to override. Reduce to `256` or `128` if you hit VRAM limits. tile-pad controls how many pixels of overlap context each tile borrows from its neighbours — the default of 64 is a good balance; reduce to `--tile-pad 32` to save VRAM.

> **Note on `-t 0` (full-frame):** Full-frame inference skips tiling entirely and processes the whole frame in one shot. This is faster for RRDB-based models (`nomos8k`, `realesrgan`, `lsdir`, `ultrasharp`) on ≤720p content with enough VRAM. However it is **not suitable for transformer models** (`nomos8kdat`, `hat`) — attention mechanisms make full-frame inference extremely slow and can produce NaN artefacts in float16 mode. Only use `-t 0` when explicitly running an RRDB model.

### --resume

The upscaler writes each frame as a PNG to a temp folder as it goes. If a run is interrupted (disk full, crash, etc.), rerun the exact same command with `--resume` added and it will skip any frames already on disk, picking up where it left off.

### --full-precision

Uses float32 instead of float16 for model inference. The quality difference is marginal in practice. Only relevant if you notice specific precision-related artefacts. Uses roughly 2x the VRAM.

### --sharpen

Adds `unsharp=3:3:0.5:3:3:0.0` to the FFmpeg encode step. Gives a crisper look on soft or lightly upscaled content. Avoid on already-sharp sources.

### --quality

Controls the HEVC output encode quality (CRF). Output is always HEVC 10-bit in an MKV container.

## Examples

```bash
# Standard compressed broadcast rip
./upscale_video.sh -i recording.mkv -r 1080p

# Heavily degraded source — maximum pre-processing
./upscale_video.sh -i old_vhs.mkv -r 1080p --prefilter heavy --deinterlace

# Clean source — highest quality model, no pre-processing
./upscale_video.sh -i bluray_rip.mkv -r 2160p -m hat --prefilter none

# Low VRAM — reduce tile size
./upscale_video.sh -i video.mkv -r 1080p -t 256

# Resume an interrupted run
./upscale_video.sh -i recording.mkv -r 1080p --resume

# Custom output name, add sharpening
./upscale_video.sh -i film.mkv -r 2160p -o film_4k.mkv --sharpen

# Batch process a folder
for f in /path/to/videos/*.mkv; do
    ./upscale_video.sh -i "$f" -r 1080p
done
```

## Monitoring a Run

```bash
# Terminal 1: run the upscale
./upscale_video.sh -i film.mkv -r 1080p

# Terminal 2: watch GPU utilisation (expect 80-100%)
watch -n 1 nvidia-smi

# Terminal 3: watch disk space
watch -n 5 df -h
```

Disk usage during a run: the temp folder (`~/ai-upscale/temp/`) holds the upscaled PNG frames at the **target** resolution. Frame size scales with output pixels — check your free space before starting a long run.

| Output resolution | ~MB per frame | 2-hour film (24fps) |
|-------------------|--------------|---------------------|
| 1080p             | ~3 MB        | ~500 GB             |
| 1440p             | ~6 MB        | ~1 TB               |
| 2160p (4K)        | ~12 MB       | **~2 TB**           |

> **Warning:** A 2-hour 1080p→2160p upscale needs approximately 2TB of free space for temp frames. Ensure the partition hosting `~/ai-upscale/temp/` has enough room before starting. Use `df -h` to check. If a run runs out of disk space mid-way, free space and resume with `--resume`.

## Testing the Installation

```bash
cd ~/ai-upscale
./test.sh
```

## Directory Structure

```
~/ai-upscale/
├── upscale_video.sh        # Main upscaling script
├── test.sh                 # Installation test script
├── venv/                   # Python virtual environment
├── models/                 # AI model files (.pth)
│   ├── 4xNomos8kDAT.pth             ← default model
│   ├── 4xNomos8kSC.pth
│   ├── 4xLSDIR.pth
│   ├── 4x-UltraSharp.pth
│   ├── RealESRGAN_x4plus.pth
│   └── HAT-L_SRx4_ImageNet-pretrain.pth
└── temp/                   # Temporary processing files (auto-created and deleted)
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
cd ~/ai-upscale
source venv/bin/activate
python3 -c "import torch; print(torch.cuda.is_available())"
# If False: reinstall PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### spandrel not found
```bash
source ~/ai-upscale/venv/bin/activate
pip install spandrel spandrel-extra-arches
```

### Out of VRAM
```bash
# Reduce tile size
./upscale_video.sh -i video.mkv -r 2160p -t 256
# Or smaller still
./upscale_video.sh -i video.mkv -r 2160p -t 128
```

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
- spandrel + spandrel-extra-arches
- opencv-python, numpy, tqdm

## Credits

- spandrel: github.com/chaiNNer-org/spandrel
- Real-ESRGAN: github.com/xinntao/Real-ESRGAN
- HAT: github.com/XPixelGroup/HAT
- 4xNomos8kSC / 4xLSDIR: github.com/Phhofm/models
- 4x-UltraSharp: huggingface.co/Kim2091
- FFmpeg: ffmpeg.org
- PyTorch: pytorch.org
