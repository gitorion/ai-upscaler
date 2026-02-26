# AI Video Upscaler

AI-powered video upscaling for Ubuntu with NVIDIA GPU acceleration. Built around the spandrel model loader with a focus on live-action content — compressed, noisy, and artifact-heavy sources.

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
| `lsdir` | `4xLSDIR.pth` | Sharp detail, real-world degradations | openmodeldb.info |
| `ultrasharp` | `4x-UltraSharp.pth` | Maximum sharpness on cleaner sources | huggingface.co/Kim2091/UltraSharp |
| `realesrgan` | `RealESRGAN_x4plus.pth` | Legacy fallback | github.com/xinntao/Real-ESRGAN/releases |
| `hat` | `HAT-L_SRx4_ImageNet-pretrain.pth` | Highest fidelity, clean sources only | github.com/XPixelGroup/HAT/releases |

All models are 4x. The script uses the model's native 4x output and resizes to your target resolution at the FFmpeg encode step.

Note: `hat` additionally requires `pip install spandrel-extra-arches` in the venv.

## Basic Usage

```bash
cd ~/ai-upscale

# Upscale to 1080p (uses nomos8k + medium prefilter by default)
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
  -m, --model TYPE          nomos8k (default), lsdir, ultrasharp, realesrgan, hat

Pre-processing:
  --prefilter LEVEL         none, light, medium (default), heavy
  --deinterlace             Deinterlace source before upscaling

Output:
  -o, --output FILE         Output filename (default: INPUT_upscaled_RES.mkv)
  -q, --quality LEVEL       high (default, crf 16), medium (crf 20), low (crf 24)
  --sharpen                 Apply unsharp mask to final output

Performance / quality:
  -t, --tile SIZE           Tile size (default: 512 — reduce to 256/128 on low VRAM)
  --tile-pad SIZE           Tile overlap padding (default: 32)
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
| `none` | Raw input — no filtering |
| `light` | Mild temporal denoise (hqdn3d) |
| `medium` | Denoise + deblock — recommended for most compressed sources |
| `heavy` | Strong denoise + deblock + deringing — for badly degraded sources |

### --deinterlace

Prepends a yadif deinterlace pass before the prefilter. Use for interlaced sources — old TV recordings, broadcast captures, VHS rips. Note: yadif mode=1 outputs a frame per field, which doubles the frame count and framerate of the source.

### --model

All models are 4x. Selection guide:

- **nomos8k** — trained on real-world degradations, handles compression well. Best starting point for most sources.
- **lsdir** — tends to produce sharper edges and finer detail on detailed scenes.
- **ultrasharp** — maximum perceived sharpness. Can over-sharpen on already-noisy sources.
- **realesrgan** — the original RealESRGAN x4plus. Kept as a fallback if others aren't downloaded.
- **hat** — Hybrid Attention Transformer. Highest fidelity model but designed for clean (lightly degraded) sources. Pair with `--prefilter light` or `none`.

### --tile / --tile-pad

Each frame is split into tiles for GPU processing. Larger tiles give slightly better quality at tile borders but require more VRAM. tile-pad controls how many pixels of overlap context each tile borrows from its neighbours — increasing this (e.g. `--tile-pad 64`) reduces visible seam lines on high-contrast content at the cost of extra VRAM.

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

# Cleaner source — highest quality model, lighter pre-processing
./upscale_video.sh -i bluray_rip.mkv -r 2160p -m hat --prefilter light

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

Disk usage during a run: the temp folder (`~/ai-upscale/temp/`) holds the upscaled PNG frames. For a 480p→1080p upscale these are ~3MB each. A 2-hour film at 24fps is ~170,000 frames — around 500GB. Ensure the partition hosting the temp folder has enough space before starting a long run.

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
