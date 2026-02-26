# Quick Start Guide

## 1. Install

```bash
chmod +x install.sh
./install.sh

# If a reboot was required for the GPU driver:
./install.sh --resume
```

## 2. Download Models

Place `.pth` files in `~/ai-upscale/models/`. Start with nomos8k — it's the default and best all-rounder for compressed live-action.

```bash
mkdir -p ~/ai-upscale/models
cd ~/ai-upscale/models

# nomos8k — recommended default (download from openmodeldb.info)
# Search: "4xNomos8kSC" → download 4xNomos8kSC.pth

# ultrasharp — direct download
wget https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth

# realesrgan — direct download (legacy fallback)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# lsdir — download from openmodeldb.info
# Search: "4xLSDIR" → download 4xLSDIR.pth

# hat — download from github.com/XPixelGroup/HAT/releases
# Requires: pip install spandrel-extra-arches
```

At minimum you need one model downloaded. nomos8k is the best starting point.

## 3. Verify Installation

```bash
cd ~/ai-upscale
./test.sh
```

All 7 checks should pass. The most important are PyTorch CUDA (check 5) and spandrel (check 6).

## 4. Upscale Your First Video

```bash
cd ~/ai-upscale

# Basic — uses nomos8k + medium prefilter by default
./upscale_video.sh -i /path/to/video.mkv -r 1080p
```

Output will be saved as `video_upscaled_1080p.mkv` in the current directory.

---

## Common Use Cases

### Compressed broadcast recording or web download
```bash
./upscale_video.sh -i recording.mkv -r 1080p
```
Uses default settings: nomos8k model, medium prefilter (denoise + deblock), tile 512, float16.

### Heavily degraded or old recording
```bash
./upscale_video.sh -i old_recording.mkv -r 1080p --prefilter heavy
```

### Interlaced source (old TV capture, VHS)
```bash
./upscale_video.sh -i vhs_capture.mkv -r 1080p --prefilter heavy --deinterlace
```

### Cleaner source — push for higher quality
```bash
./upscale_video.sh -i bluray_rip.mkv -r 2160p -m hat --prefilter light
```

### Low VRAM (under 8GB)
```bash
./upscale_video.sh -i video.mkv -r 1080p -t 256
```

### Resume an interrupted run
```bash
# Use the exact same command as before, add --resume
./upscale_video.sh -i recording.mkv -r 1080p --resume
```

### Upscale to 4K with sharpening
```bash
./upscale_video.sh -i film.mkv -r 2160p --sharpen
```

### Custom output path
```bash
./upscale_video.sh -i film.mkv -r 1080p -o /mnt/storage/film_upscaled.mkv
```

---

## Choosing a Model

| Source type | Recommended model | Prefilter |
|-------------|-------------------|-----------|
| Compressed web download / broadcast | `nomos8k` (default) | `medium` (default) |
| Very noisy or heavily compressed | `nomos8k` or `lsdir` | `heavy` |
| Reasonably clean (DVD/BluRay rip) | `lsdir` or `hat` | `light` |
| Maximum sharpness on clean source | `ultrasharp` or `hat` | `light` or `none` |

---

## Choosing a Prefilter

The prefilter runs an FFmpeg cleaning pass before the AI processes the frame. Cleaning compression artifacts first leads to sharper, more accurate upscaling.

| Level | When to use |
|-------|-------------|
| `none` | Source is already clean (lossless or near-lossless) |
| `light` | Mild compression, low noise |
| `medium` | Typical compressed source — streaming, web download, broadcast rip |
| `heavy` | Heavy blocking, ringing, or noise — old VHS, heavily re-encoded files |

---

## Monitoring Disk Space

Temp PNG frames are written to `~/ai-upscale/temp/` during processing. For long videos these can be large — check available space before starting:

```bash
df -h ~/ai-upscale/

# Watch during a run
watch -n 10 df -h
```

If a run stops with `libpng error: Write Error`, the disk is full. Free space and rerun with `--resume`.

---

## Batch Processing

```bash
#!/bin/bash
for video in /path/to/videos/*.mkv; do
    echo "Processing: $video"
    ~/ai-upscale/upscale_video.sh -i "$video" -r 1080p
done
```

---

## Useful Commands

```bash
# Check video info before upscaling
ffprobe -v error -show_streams -select_streams v:0 input.mkv

# Check GPU status
nvidia-smi

# Monitor GPU during processing
watch -n 1 nvidia-smi

# Verify spandrel is installed
source ~/ai-upscale/venv/bin/activate
python3 -c "import spandrel; print('spandrel OK')"

# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
