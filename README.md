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

The installer downloads BasicVSR++ and SPyNet weights automatically. All single-frame models must be downloaded manually to `~/ai-upscale/models/`.

### Single-frame models (spandrel)

Process one frame at a time. No additional dependencies beyond the base install.

**SPAN — fastest (~0.5–1 s/frame)**

| Key | Filename | Best for | Speed (4060 Ti 16GB) |
|-----|----------|----------|----------------------|
| `spanmedium` | `4xNomos8k_span_otf_medium.pth` | Real-world degraded sources — best fast all-rounder | ~1 s/frame @ 480p |
| `spanweak` | `4xNomos8k_span_otf_weak.pth` | Light degradation — better/cleaner sources | ~1 s/frame @ 480p |
| `spanstrong` | `4xNomos8k_span_otf_strong.pth` | Heavy degradation — badly compressed sources | ~1 s/frame @ 480p |

**RealPLKSR — fast (~1–2 s/frame)**

| Key | Filename | Best for | Speed (4060 Ti 16GB) |
|-----|----------|----------|----------------------|
| `webphoto` | `4xNomosWebPhoto_RealPLKSR.pth` | Web/streaming: lens blur + JPEG/WebP + noise | ~2 s/frame @ 480p |
| `nomos2plksr` | `4xNomos2_realplksr_dysample.pth` | Cleaner compressed sources, less aggressive | ~2 s/frame @ 480p |

**RRDB — standard (~4 s/frame)**

| Key | Filename | Best for | Speed (4060 Ti 16GB) |
|-----|----------|----------|----------------------|
| `nomos8k` | `4xNomos8kSC.pth` | Compressed live-action — **default** | ~4 s/frame @ 480p |
| `lsdirplus` | `4xLSDIRplus.pth` | LSDIR dataset + real degradation — sharp detail on degraded sources | ~4 s/frame @ 480p |
| `lsdir` | `4xLSDIR.pth` | LSDIR dataset — sharp detail on clean sources | ~4 s/frame @ 480p |
| `ultrasharp` | `4x-UltraSharp.pth` | Maximum sharpness on clean sources | ~4 s/frame @ 480p |
| `realesrgan` | `RealESRGAN_x4plus.pth` | Legacy fallback | ~4 s/frame @ 480p |

**Transformer — slow (20–60 s/frame, short clips only)**

| Key | Filename | Best for | Speed (4060 Ti 16GB) |
|-----|----------|----------|----------------------|
| `atdjpg` | `4xNomos8k_atd_jpg.pth` | Best for heavily JPEG-compressed/degraded sources | ~6 s/frame @ 480p |
| `nomos8kschat` | `4xNomos8kSCHAT-L.pth` | HAT-L quality on real-world/compressed sources | ~20 s/frame @ 480p |
| `hat` | `HAT-L_SRx4_ImageNet-pretrain.pth` | Highest fidelity transformer, clean sources only | ~20 s/frame @ 480p |
| `nomos8kdat` | `4xNomos8kDAT.pth` | DAT transformer — highest quality, short clips only | ~24 s/frame @ 480p |

Note: `hat` additionally requires `pip install spandrel-extra-arches` in the venv.

#### Download commands

```bash
cd ~/ai-upscale/models

# ── SPAN (~0.5-1 s/frame) ────────────────────────────────────────────────────

# spanmedium — 8.6 MB
wget -O 4xNomos8k_span_otf_medium.pth \
  "https://drive.usercontent.google.com/download?id=1M1bgiMOuOoZkArB-JkKoZiDNoHlpar2X&export=download&confirm=t"

# spanweak — 8.6 MB
wget -O 4xNomos8k_span_otf_weak.pth \
  "https://drive.usercontent.google.com/download?id=1KkV23QH3oBAtl88HBKMq88cdWEv0J2hV&export=download&confirm=t"

# spanstrong — 8.6 MB
wget -O 4xNomos8k_span_otf_strong.pth \
  "https://drive.usercontent.google.com/download?id=1K_OUt9lwvDXn280OTfURm0tD3E0lVZ5b&export=download&confirm=t"

# ── RealPLKSR (~1-2 s/frame) ─────────────────────────────────────────────────

# webphoto — 28 MB
wget -O 4xNomosWebPhoto_RealPLKSR.pth \
  "https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth"

# nomos2plksr — 28 MB
wget -O 4xNomos2_realplksr_dysample.pth \
  "https://github.com/Phhofm/models/releases/download/4xNomos2_realplksr_dysample/4xNomos2_realplksr_dysample.pth"

# ── RRDB (~4 s/frame) ────────────────────────────────────────────────────────

# nomos8k (default) — 64 MB
wget -O 4xNomos8kSC.pth \
  "https://github.com/Phhofm/models/releases/download/4xNomos8kSC/4xNomos8kSC.pth"

# lsdirplus — 64 MB (LSDIR + real degradation)
wget -O 4xLSDIRplus.pth \
  "https://github.com/Phhofm/models/raw/main/4xLSDIRplus/4xLSDIRplus.pth"

# ultrasharp — 64 MB
wget -O 4x-UltraSharp.pth \
  "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth"

# lsdir — 64 MB (no direct link — download via browser)
# Visit: openmodeldb.info/models/4x-LSDIR — place 4xLSDIR.pth in ~/ai-upscale/models/

# realesrgan — 64 MB
wget -O RealESRGAN_x4plus.pth \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# ── Transformer (20-60 s/frame, short clips only) ────────────────────────────

# atdjpg — 78 MB
wget -O 4xNomos8k_atd_jpg.pth \
  "https://github.com/Phhofm/models/releases/download/4xNomos8k_atd_jpg/4xNomos8k_atd_jpg.pth"

# nomos8kschat — 316 MB
wget -O 4xNomos8kSCHAT-L.pth \
  "https://drive.usercontent.google.com/download?id=1gh7HDKzf9aZw-rA8WYQy1ZZ8D0MAIHxR&export=download&confirm=t"

# hat — 158 MB (requires: pip install spandrel-extra-arches)
wget -O HAT-L_SRx4_ImageNet-pretrain.pth \
  "https://huggingface.co/anchuang/HAT-L_SRx4_ImageNet-pretrain/resolve/main/HAT-L_SRx4_ImageNet-pretrain.pth"

# nomos8kdat — 147 MB
wget -O 4xNomos8kDAT.pth \
  "https://github.com/Phhofm/models/releases/download/4xNomos8kDAT/4xNomos8kDAT.pth"
```

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

### Diffusion VSR (SeedVR2)

`seedvr2` is a different class of model — **one-step diffusion video super-resolution**. Where the spandrel/basicsr models *sharpen and clean* the detail already present, SeedVR2 *reconstructs* plausible detail using a learned prior of how real video looks. On genuinely low-resolution, compressed live-action it is the single biggest quality jump available — well beyond what ESRGAN/transformer/BasicVSR++ can reach.

It comes with trade-offs you must weigh against your use case:

- **Slow.** ~4–7 s/frame on a 4060 Ti 16GB. Practical for short, important clips — **not** full-length episodes (a 45-min episode would take days). For length, BasicVSR++ remains the workhorse.
- **Generative.** It can *fabricate* fine detail that wasn't in the source (skin texture, small text). Usually convincing, occasionally a fidelity drift — a real tension with faithful archival restoration. Judge it on your own content.
- **Separate install.** SeedVR2 needs a different PyTorch build, so it runs in its **own venv** (`~/ai-upscale/seedvr2/`) via the upstream standalone CLI — it does not touch the main pipeline's venv. Set it up once:

  ```bash
  cd <repo>/prototype/seedvr2 && ./setup.sh
  ```

  Model weights (3B FP8, ~3.6 GB) auto-download to `~/ai-upscale/models/SEEDVR2/` on first run — nothing to fetch manually.

Once installed, it's a first-class model key — the main script handles audio, subtitles, and aspect-ratio correction exactly as with any other model:

```bash
./upscale_video.sh -i clip.mkv -r 1080p -m seedvr2
```

Aspect ratio is corrected **up front** (anamorphic sources are de-anamorphized to square pixels before SeedVR2 sees them, so detail is reconstructed at correct geometry rather than stretched afterwards). The final step then **stream-copies** SeedVR2's video and only muxes on audio/subtitles — no second encode, no generation loss. (Passing `--sharpen` is the one exception: a filter forces a re-encode.)

**Motion-quality defaults (batch size).** `SEEDVR2_BATCH` is the single most important setting for movement. Each batch generates its detail independently, so the batch length *is* the window of temporal context. At the old default of 13 (~0.5s at 25fps) consecutive batches disagree during motion — the model re-invents texture on moving surfaces, which reads as crawling/shimmer — while static shots, where successive batches see near-identical input, look excellent. That "great on stills, poor on movement" split is the signature of too small a batch. The default is now **21** (~0.85s), funded by raising `SEEDVR2_BLOCKS_SWAP` to 32 (its max for the 3B model) to free the VRAM. `SEEDVR2_CHUNK` is 252 = 21 × 12 so every batch in a chunk is full-length — a ragged final batch is exactly what `temporal_overlap` is documented to compensate for. Expect a **slower** run: more block swapping means more CPU↔GPU traffic. Batch must always be `4n+1` (1, 5, 9, 13, 17, 21, 25…); 25 is the next step if you have headroom.

**Automatic OOM recovery.** If a run runs out of VRAM, SeedVR2 no longer just dies — it steps down to a config that fits, cheapest-quality-cost first. `blocks_to_swap` is raised to its maximum before anything else, because offloading more transformer blocks to CPU RAM costs only speed and leaves the output bit-identical. Only when that is exhausted does it reduce `batch_size` down the 4n+1 ladder (21 → 17 → 13 → 9 → 5), warning each time that the temporal window is shrinking — which is precisely the thing that causes motion shimmer, so it is the last resort. `chunk_size` is realigned to a whole multiple of the new batch automatically, and the surviving config is reused for the rest of the run so later segments don't each re-pay a failed attempt. Note this differs from FlashVSR's recovery, which bisects the clip: that would achieve nothing here because SeedVR2 already streams via `chunk_size`, so its peak VRAM is set by batch size and resolution rather than clip length.

**Defaults tuned from real-content A/B testing:** SeedVR2 uses `--prefilter light` and `SEEDVR2_TEMPORAL_OVERLAP=0`. On flat surfaces (e.g. a static wall) during motion, the generative model can produce two distracting artifacts: a shimmering **noise layer** (it hallucinates unstable texture from source grain) and a **ghost/echo** that doesn't track the scene. A light pre-denoise tames the noise layer (giving the model a cleaner canvas), and overlap `0` removes the ghost (which comes from blending two batches' differing generations across the overlap). Both were confirmed to help on a real 480p source. Tune per source if needed: drop to `--prefilter none` to feed maximum detail (cleaner sources), step up to `medium` for heavy compression blocking, or raise `SEEDVR2_TEMPORAL_OVERLAP` if you ever see chunk-boundary seams instead of ghosting.

**Long files (auto-segmentation + resume):** A full episode is ~4–5 days of compute, and the SeedVR2 CLI has no mid-run checkpoint — a single crash or reboot would lose everything. So files longer than `SEEDVR2_SEGMENT_SECONDS` (default 5 min) are automatically split into segments, each upscaled independently and written atomically, then losslessly concatenated with audio/subs muxed back on. If a run is interrupted, re-run the **exact same command with `--resume`** and it skips every already-finished segment and picks up where it left off. At completion the segment intermediates are cleaned up and you get a single output file. So a 45-minute source *is* practical — it just takes days, survives interruptions, and you can stop/resume freely. Set `SEEDVR2_SEGMENT_SECONDS=0` to force single-shot.

**Whole seasons (`batch_upscale.sh`):** to upscale a folder of episodes in one unattended run, use the batch wrapper instead of calling `upscale_video.sh` per file:

```bash
./batch_upscale.sh -i ~/season_1 -r 1080p --prefilter none -m seedvr2
```

It processes each episode in turn, forwarding every upscaler flag verbatim (so quality is identical to a single run), and is resumable at **two levels**:

- **Between episodes** — an episode whose finished output already exists (and is a valid video) is skipped. So if 3 of 9 episodes are done and the run is interrupted, re-running the *same command* picks up at episode 4. There's no state file to lose: the finished outputs themselves are the record of progress.
- **Within an episode** — each child run is invoked with `--resume`, so an episode that was interrupted mid-way (its segments survive under `temp/<episode>/`) continues rather than restarting.

Disk stays flat across the season: the child cleans up its own temp when each episode finishes, so only one episode's working files exist at a time alongside the completed outputs. Finished files land in a sibling folder named `<input>_<res>` by default — e.g. `~/season_1` → `~/season_1_1080p/` (override with `-o`) — named `<episode>_upscaled_<res>.mkv`. By default the batch **stops on the first failed episode** and reports it (so a systemic problem like OOM or a full disk doesn't silently burn days across every episode) — pass `--keep-going` to continue past failures, or `--force` to reprocess episodes that already have outputs (**overwrites them**). Because it's non-interactive, `-r` is required.

**Disk for a long run** (≈45 min, 720p→1080p): budget **~60 GB free** with the default prefilter, or **~20 GB** with `--prefilter none`. The big cost is the lossless FFV1 prefilter intermediate (~20 GB) plus the input segments being lossless copies of it (another ~20 GB) — `--prefilter none` removes the FFV1 entirely (and SeedVR2's diffusion handles degradation well on its own), cutting disk ~3×. Rule of thumb at 1080p: output side ≈ 2 MB/s × duration with ~3× headroom; FFV1 prefilter ≈ 7.5 MB/s × duration with ~2×; multiply output-side figures by ~4 for 4K. Check with `df -h ~/ai-upscale` before starting.

**16GB notes:** SeedVR2 is tuned in `upscale_video.sh` (the `SEEDVR2_*` variables) for a 4060 Ti 16GB — 3B FP8, block-swap, VAE tiling, and **chunked streaming** (`SEEDVR2_CHUNK`). The streaming is important: loading a whole clip holds the entire output in system RAM and will OOM-kill on long clips, so the default processes in bounded chunks. **32 GB system RAM recommended** (16 GB is marginal because the CPU-offload that frees VRAM lands in RAM).

**Speed (lossless only):** `torch.compile` is on by default (`SEEDVR2_COMPILE`) — it fuses GPU kernels via Triton for a 20–40% speedup without changing the output. Attention defaults to `auto` (`SEEDVR2_ATTENTION`): if `flash-attn` is installed in the SeedVR2 venv it uses `flash_attn_2` (lossless, faster) automatically, otherwise it falls back to `sdpa` — so the only step to get the speedup is `pip install flash-attn`, no flag to flip. SageAttention is deliberately **not** used — it quantizes attention (an approximation), which conflicts with the quality-first goal.

### Streaming Diffusion VSR (FlashVSR)

`flashvsr` is the second diffusion engine, added alongside SeedVR2 rather than replacing it — both remain fully supported and you pick per source.

**Why it exists.** SeedVR2 generates in fixed batches (`4n+1` frames). Each batch produces its detail independently, so during **motion** consecutive batches can disagree — crawling/shimmering texture, and discontinuities at batch boundaries — while static shots look excellent. That "great on stills, poor on movement" split is the characteristic SeedVR2 failure. [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) (CVPR 2026) is built around *streaming* inference instead, specifically targeting temporal stability.

Neither engine dominates. Published comparisons put SeedVR2 ahead on short, heavily-compressed clips and FlashVSR ahead on longer real-footage material at HD input — so on a clean 720p episode, try FlashVSR first.

**Install** (its own venv, like SeedVR2 — the main venv is untouched):

```bash
cd ~/ai-upscaler/prototype/flashvsr && ./setup.sh
```

This is the slowest setup: Block-Sparse-Attention compiles from source (10–40 min). It auto-detects your GPU's CUDA architecture — worth noting because upstream's documented arch list (`80;90;100`) **omits `89`**, which is exactly what Ada cards like the 4060 Ti are.

**Usage** is identical to any other model, including auto-segmentation, `--resume`, and `batch_upscale.sh`:

```bash
./upscale_video.sh -i s01e01.mkv -r 1080p --prefilter none -m flashvsr
```

**Output quality note.** FlashVSR's own writer is hardcoded to 8-bit H.264 CRF 20 (preset veryfast) even at its maximum `--quality 10` — which would cap this pipeline before our encoder ran. `setup.sh` patches in an opt-in **10-bit x265** writer (`FLASHVSR_OUT_CRF`, default 12 — visually transparent) so the path has exactly one high-quality encode, then muxes losslessly. The patch is idempotent, reversible, and fail-soft: if upstream refactors, it declines to apply and you fall back to upstream's 8-bit output with a warning. Details in [`prototype/flashvsr/README.md`](prototype/flashvsr/README.md).

**VRAM and segment length.** FlashVSR's `infer.py` builds the **entire clip** as a GPU tensor (scaled and padded to a multiple of 128) *before* loading the model, so peak VRAM scales with clip **length** — and neither `--tile-dit` nor `--tile-vae` helps, since those tile the model rather than this tensor. A 4-minute 720p clip is ~135 GB of tensor and OOMs immediately. Segment length is therefore derived from real frame geometry rather than a fixed duration that would OOM at 4K and waste capacity at 480p. It self-tunes in both directions: `FLASHVSR_INPUT_BUDGET_MB=auto` (default) probes free VRAM at run start and claims everything past the model reserve — ~19s segments on a 16GB card, ~39s on 24GB — and because that probe is deliberately optimistic, any segment that still OOMs is **bisected and retried** rather than failing the run, with only that segment paying the cost. The chosen length is persisted so `--resume` reuses the identical split. See [`prototype/flashvsr/README.md`](prototype/flashvsr/README.md) for the full mechanism.

**Scale handling and supersampling.** FlashVSR takes a scale *multiplier* rather than a target height, so `FLASHVSR_SCALE=auto` (default) derives what's needed and clamps it up to `FLASHVSR_MIN_SCALE`. That floor defaults to **1** (no supersampling): exact target size and a lossless mux. Setting it to `2.0` generates 720p at 1440p and downscales, which keeps the model nearer its trained 4× regime and **attenuates shimmer** — per-frame hallucination is high-frequency and uncorrelated between frames, exactly what downscaling averages away (the principle behind supersampled anti-aliasing). It's off by default because it makes each frame ~1.8× larger in VRAM, which on a 16GB card roughly halves the segment length. If dimensions ever miss the target, the encoder detects it and resizes rather than shipping the wrong resolution.

> **Status:** built against a close read of upstream's source and tested locally for segmentation, resume, dimension handling and the HQ writer — but **not yet run end-to-end on a GPU**. Smoke-test a short clip before a long run.

### Estimated total upscale time (1 hour, 25fps, 4060 Ti 16GB)

| Model | 480p source | 720p source |
|-------|------------|------------|
| basicvsr | ~4 hrs | ~51 hrs |
| nomos8k / ultrasharp / lsdir | ~100 hrs | ~250 hrs |
| hat | ~200 hrs | ~500 hrs |
| nomos8kdat | ~600 hrs | ~1600 hrs |
| seedvr2 | ~4–5 days (~4–7 s/frame; auto-segmented + resumable) | longer |

BasicVSR++ is 5-15x faster than single-frame models and is the recommended choice for any content longer than a few minutes. For short clips where maximum quality matters most, `seedvr2` is the top tier.

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

# Batch a whole folder / season (resumable — see below)
./batch_upscale.sh -i ~/season_1 -r 1080p --prefilter none -m seedvr2
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
  -m, --model TYPE          SPAN: spanmedium (fast default), spanweak, spanstrong
                            RealPLKSR: webphoto, nomos2plksr
                            RRDB: nomos8k (default), lsdirplus, lsdir, ultrasharp, realesrgan
                            Transformer: atdjpg, nomos8kschat, hat, nomos8kdat
                            Temporal: basicvsr
                            Diffusion VSR: seedvr2, flashvsr (separate installs)

Pre-processing:
  --prefilter LEVEL         none, light (default), medium, heavy
  --deinterlace             Deinterlace source before upscaling
                            (auto-detects interlaced vs progressive)

Output:
  -o, --output FILE         Output file (default: INPUT_upscaled_RES.mkv,
                            written next to the input, not the current directory)
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

**SPAN (~0.5–1 s/frame) — fastest single-frame tier:**

- **spanmedium** — SPAN architecture trained on Nomos8k with medium OTF degradation (resize, blur, JPEG compression). Best fast all-rounder — directly comparable to `nomos8k` in quality at 4-8x the speed. First choice for long-form content where RRDB is too slow.
- **spanweak** — Same architecture, lighter degradation. Better for sources that are compressed but not severely damaged.
- **spanstrong** — Same architecture, heavier degradation. Best for badly compressed or heavily artifacted sources when speed is still required.

**RealPLKSR (~1–2 s/frame) — fast, real-world trained:**

- **webphoto** — RealPLKSR trained on Nomos-v2 with the full realistic degradation pipeline: lens blur, realistic noise (LUDVAE), JPEG and WebP re-compression down to quality 40. Specifically targets the web/streaming download chain. 2-4x faster than RRDB.
- **nomos2plksr** — RealPLKSR on Nomos-v2, lighter JPEG-only degradation. Better for sources that are compressed but otherwise clean. Slightly less aggressive than `webphoto`.

**RRDB (~4 s/frame) — standard single-frame:**

- **nomos8k** — RRDB-based, trained on real-world degradations. Fast and reliable. Best all-rounder for compressed video. **Default.**
- **lsdirplus** — RRDB trained on the large LSDIR dataset (85K images) with real-world degradation (compression + noise + blur). Better than `lsdir` for degraded sources — same speed.
- **lsdir** — RRDB trained on the large LSDIR dataset, clean/bicubic training. Best for clean sources where fine detail preservation is the priority.
- **ultrasharp** — Maximum perceived sharpness on clean sources. Can over-sharpen noisy inputs.
- **realesrgan** — The original RealESRGAN x4plus. Kept as a legacy fallback.

**Transformer (20–60 s/frame) — short clips only:**

- **atdjpg** — ATD transformer trained on Nomos8k with aggressive JPEG compression (down to quality 40), re-compression, blur, and resizes. Best single-frame model for heavily compressed/degraded sources — DVD rips, old web downloads, heavily transcoded files. Preserves film grain. Auto full-precision, auto tile probing.
- **nomos8kschat** — HAT-L fine-tuned on Nomos8k with real-world JPEG/blur degradation. Brings HAT-L transformer quality to compressed sources rather than clean bicubic ones. Direct quality ceiling over `nomos8k` when speed is not a constraint. Auto full-precision, auto tile probing.
- **hat** — Hybrid Attention Transformer trained on clean ImageNet. Highest fidelity for clean sources. Pair with `--prefilter none`.
- **nomos8kdat** — DAT transformer. Highest single-frame quality available — very slow, short clips only.

**Temporal (basicsr) — process a sliding window of frames:**

- **basicvsr** — BasicVSR++ with bidirectional propagation and optical flow alignment. 5-15x faster than single-frame models. Better temporal consistency and less flickering. The only practical model for full-length TV episodes and movies on consumer hardware. Best for degraded/compressed sources where noise is consistent across frames.

**Diffusion VSR — highest quality on low-res sources, short clips only:**

- **seedvr2** — One-step diffusion VSR. *Reconstructs* plausible detail rather than only sharpening, making it the biggest quality jump available on genuinely low-res/compressed live-action. Slow (~4–7 s/frame on 16GB) and generative (can fabricate fine detail). Long files are auto-segmented and resumable (`--resume`), so full episodes are practical given the time — but for length BasicVSR++ is far faster. Separate install (`prototype/seedvr2/setup.sh`) — see the [Diffusion VSR (SeedVR2)](#diffusion-vsr-seedvr2) section above for the full rundown and 16GB notes.

### --temporal-window

Controls the sliding window size for temporal models. Defaults to **auto** — after loading the model, the script probes GPU VRAM by running a test window with descending sizes (15 → 13 → 11 → 9 → 7 → 5 → 3) and selects the largest that fits. A larger window gives better temporal continuity for slow pans and smooth motion. Pass `--temporal-window N` to override.

### --tile / --tile-pad

Tile size defaults to **auto** — after loading the model, the script probes GPU VRAM by running a test frame with descending tile sizes (full-frame → 768 → 512 → 384 → 256 → 192 → 128) and selects the largest size that fits. This automatically adapts to any GPU, model, and precision combination.

Pass `-t SIZE` to override with a specific value. Tile-pad controls how many pixels of overlap context each tile borrows from its neighbours — the default of 64 is a good balance. There is no quality loss from tiling.

### --resume

The upscaler writes each frame as a PNG to a temp folder as it goes. If a run is interrupted, rerun the exact same command with `--resume` added and it will skip any frames already on disk, picking up where it left off.

### --full-precision

Uses float32 instead of float16 for model inference. Auto-enabled for transformer models (`hat`, `nomos8kdat`, `atdjpg`, `nomos8kschat`) which overflow float16. Not needed for SPAN or RealPLKSR models — they run correctly in float16. For RRDB models the quality difference is marginal. Uses roughly 2x the VRAM — the auto tile probe accounts for this.

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
