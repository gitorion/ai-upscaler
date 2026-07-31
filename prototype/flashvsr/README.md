# FlashVSR kit

Installs [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) (CVPR 2026) as a first-class
`-m flashvsr` engine in `upscale_video.sh`, alongside — not replacing — SeedVR2.

We install the [FlashVSR-Pro](https://github.com/LujiaJin/FlashVSR-Pro) fork rather than upstream,
because it ships a real standalone `infer.py` (upstream only has per-mode example scripts), plus
DiT/VAE tiling for low-VRAM cards and a long-video streaming mode. Both are Apache-2.0.

## Why this engine

SeedVR2 processes fixed batches of frames (`4n+1`). Each batch generates its detail independently,
so on **motion** consecutive batches can disagree — which shows up as crawling/shimmering texture
and boundary discontinuities, while static shots look excellent. FlashVSR is built around
*streaming* inference instead, which targets exactly that failure mode.

Neither is strictly better. Reported comparisons put SeedVR2 ahead on short, heavily-compressed
clips, and FlashVSR ahead on longer real-footage material at HD input. Keep both; pick per source.

## Install (on the GPU VM)

```bash
cd ~/ai-upscaler/prototype/flashvsr
./setup.sh
```

This is the slowest of the three setups — **Block-Sparse-Attention is compiled from source**
(10–40 min). Everything lands under `~/ai-upscale/flashvsr/` in its own venv (pinned torch
2.6.0+cu124), so the main venv and the SeedVR2 venv are untouched. Weights (~several GB) go to
`~/ai-upscale/models/FLASHVSR/FlashVSR-v1.1`, outside the repo, so re-cloning never re-downloads.

**CUDA arch note:** upstream's docs suggest building for `80;90;100`, which **omits `89`** — the
architecture of Ada cards including the RTX 4060 Ti. `setup.sh` queries the GPU with `nvidia-smi`
and builds for what's actually present. Override with `BLOCK_SPARSE_ATTN_CUDA_ARCHS` if needed.

## Usage

```bash
# Short clip — smoke-test first
~/ai-upscale/upscale_video.sh -i clip.mkv -r 1080p -m flashvsr

# Full episode (auto-segmented + resumable, same as seedvr2)
~/ai-upscale/upscale_video.sh -i s01e01.mkv -r 1080p --prefilter none -m flashvsr

# A whole season
~/ai-upscaler/batch_upscale.sh -i ~/season_1 -r 1080p --prefilter none -m flashvsr
```

## The output-quality patch (important)

FlashVSR-Pro hardcodes its writer. Even at its maximum `--quality 10` it emits:

```
8-bit H.264, yuv420p, CRF 20, preset veryfast      (or NVENC QP 20, worse)
```

For a quality-first pipeline that is a hard ceiling — the diffusion result would be crushed to
8-bit CRF 20 *before* our encoder ever sees it, and nothing downstream can recover it.

`setup.sh` therefore applies `patch_hq_output.py`, which injects an opt-in early-return into
`save_video()` that writes **x265 10-bit** at a configurable CRF instead. `upscale_video.sh`
enables it per call via `AIUPSCALER_HQ_OUT=1`. The result is one high-quality encode which is then
muxed on losslessly — no second generation loss.

The patch is:

- **opt-in** — without the env var, behaviour is byte-identical to upstream;
- **idempotent** — safe to re-run;
- **reversible** — the original is kept as `infer.py.orig`;
- **fail-soft** — if upstream refactors `save_video`, the patch refuses to apply, leaves the file
  untouched, verifies it still compiles, and `setup.sh` only warns. You then get upstream's 8-bit
  output: still functional, just lower quality.

Tune with `FLASHVSR_OUT_CRF` (default `12`, visually transparent) and `FLASHVSR_OUT_PRESET`.

## Tuning

All settings are env-overridable per run (see the `FLASHVSR_*` block in `upscale_video.sh`):

| Variable | Default | Notes |
|---|---|---|
| `FLASHVSR_MODE` | `full` | `full` (best quality) · `tiny` (lower VRAM) · `tiny-long` (streaming) |
| `FLASHVSR_TILE_VAE` | `true` | Needed for `full` on 16GB |
| `FLASHVSR_TILE_DIT` | `false` | **First thing to enable if you OOM** — big VRAM saver, some seam risk |
| `FLASHVSR_TILE_OVERLAP` | `32` | Blend width between tiles (raised from upstream's 24 to soften seams) |
| `FLASHVSR_SCALE` | `auto` | Ratio needed for the target, clamped up to `MIN_SCALE` |
| `FLASHVSR_MIN_SCALE` | `2.0` | Supersampling floor — see below. `1` disables |
| `FLASHVSR_COLOR_FIX` | `true` | Corrects diffusion colour drift toward the source |
| `FLASHVSR_OUT_CRF` | `12` | The deliverable encode |
| `FLASHVSR_SEGMENT_SECONDS` | `300` | `0` disables segmentation |

### Scale and supersampling

FlashVSR takes a scale *multiplier*, not a target height, so `auto` derives what's needed
(720p→1080p = 1.5) and then clamps it up to `FLASHVSR_MIN_SCALE` (default 2.0). A fixed scale can't
be the default because it breaks across mixed sources — 2.0 on a 480p source undershoots a 1080p
target.

The 2.0 floor exists for two reasons, both aimed at motion artifacts:

1. **Supersampling.** Generating 720p at 2× (1440p) and downscaling to 1080p averages neighbouring
   pixels. Per-frame hallucination is high-frequency and uncorrelated between frames, so that
   downscale attenuates exactly the component that reads as shimmer — the same principle as
   supersampled anti-aliasing.
2. **In-distribution.** FlashVSR is trained as a 4× restorer and its own CLI defaults to 2.0.
   Running it at 1.5× is below its design point.

Costs: ~1.8× the pixels of a 1.5× run (slower, more VRAM — this is the most likely OOM cause on
16GB), and one corrective re-encode from the 10-bit intermediate instead of a stream copy, which is
near-transparent at these CRFs. If dimensions ever miss the target, `encode_output` detects it and
resizes rather than silently shipping the wrong resolution.

Set `FLASHVSR_MIN_SCALE=1` to get exact-size output and a lossless mux, without the supersampling.

## Status

Written from a close read of upstream's source (CLI flags, output-path logic, dimension handling
and writer were all verified against the actual `infer.py`), and the segmentation/resume logic and
HQ writer were tested locally. **It has not yet been run end-to-end on a GPU.** Smoke-test on a
short clip before committing to a long run.
