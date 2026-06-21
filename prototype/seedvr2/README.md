# SeedVR2 prototype kit

A **throwaway evaluation harness** for trialling SeedVR2 (one-step diffusion video
super-resolution) against the existing single-frame / BasicVSR++ pipeline on a few
representative clips. The goal is to judge — on *your own sources* — whether the
generative detail is worth the speed cost and the hallucination/flicker risk before
deciding to integrate it as a first-class `-m seedvr2` engine.

## How this fits the existing project

It's designed to **sit alongside** the current pipeline, not replace or disturb it:

- **Additive only.** Lives entirely under `prototype/seedvr2/` and creates runtime files
  under `~/ai-upscale/seedvr2/`. It changes nothing in the existing pipeline, so it can
  ride on `main` and be pushed/pulled normally without affecting production behaviour.
- **Scripts are tracked here**, in the `ai-upscaler` git clone (`prototype/seedvr2/`).
- **All runtime artifacts go under the existing `~/ai-upscale/` runtime tree**, mirroring
  how the main script already uses it:

  ```
  ~/ai-upscale/                     (existing runtime root — unchanged)
  ├── venv/                         ← main pipeline venv (UNTOUCHED)
  ├── models/
  │   ├── *.pth                     ← existing single-frame / temporal models
  │   └── SEEDVR2/                  ← SeedVR2 weights (auto-downloaded) — shares models/
  └── seedvr2/                      ← everything this kit creates, isolated here
      ├── venv/                     ← separate venv (SeedVR2 needs a different torch)
      ├── repo/                     ← upstream numz/ComfyUI-SeedVR2 checkout
      └── eval/{clips,out}/         ← test clips + outputs
  ```

  The separate SeedVR2 venv is deliberate (torch conflict), but it lives under the same
  runtime root and shares `models/`, so nothing is alien to the existing layout. The main
  `~/ai-upscale/venv` and the production pipeline are never modified.

## Where this runs

On the **GPU VM (Ubuntu + RTX 4060 Ti 16GB)** — SeedVR2 requires CUDA. On the VM the
project already exists; just pull `main` into the `ai-upscaler` clone as usual:

```bash
cd ~/ai-upscaler          # the existing git clone on the VM
git pull
cd prototype/seedvr2
./setup.sh
```

`setup.sh` clones [`numz/ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
into `~/ai-upscale/seedvr2/repo`, builds `~/ai-upscale/seedvr2/venv`, and installs torch +
requirements. Weights (3B FP8 ≈ a few GB) auto-download to `~/ai-upscale/models/SEEDVR2`
on first run.

> ⚠️ **The upstream CLI changes.** Before trusting `run.sh`, confirm flag names:
> `source ~/ai-upscale/seedvr2/venv/bin/activate && python ~/ai-upscale/seedvr2/repo/inference_cli.py -h`.
> If a flag was renamed (`--dit_model` vs `--model`), edit the variable block atop `run.sh`.

## Workflow

```bash
# 1. Cut representative clips (faces, fine texture, fast motion, heavy compression).
#    Writes to ~/ai-upscale/seedvr2/eval/clips/.
./make_clips.sh /path/to/source.mkv  00:05:30  00:21:00  00:48:15

# 2. Upscale a clip with SeedVR2 (16GB-safe defaults), target 1080p short edge.
./run.sh ~/ai-upscale/seedvr2/eval/clips/clip_01.mkv  1080

# 3. Produce the SAME clip with the current pipeline for an honest A/B.
~/ai-upscale/upscale_video.sh -i ~/ai-upscale/seedvr2/eval/clips/clip_01.mkv \
    -r 1080p -m basicvsr -o ~/ai-upscale/seedvr2/eval/out/clip_01_basicvsr.mkv

# 4. Stack them side by side.
./compare.sh ~/ai-upscale/seedvr2/eval/out/clip_01_basicvsr.mkv \
             ~/ai-upscale/seedvr2/eval/out/clip_01_seedvr2.mkv
```

(`run.sh` prints the exact commands for steps 3–4 with the right paths filled in.)

## What to look for when judging

- **Detail reconstruction** — does it invent *plausible* texture (skin, hair, fabric,
  foliage) where the current models only sharpen mush? The upside.
- **Hallucination / fidelity drift** — does it fabricate features, shift small facial
  details, or mangle small text/logos? The risk that conflicts with faithful restoration.
- **Temporal stability** — watch for flicker / shimmer / "crawling" texture on motion.
  Raising `BATCH_SIZE` in `run.sh` (4n+1: 5 → 9 → 13) improves consistency at VRAM cost.
- **Speed** — `run.sh` prints elapsed seconds; compare s/frame to BasicVSR++/nomos8k.

## 16GB tuning knobs (in `run.sh`)

| Knob | Default | Effect |
|------|---------|--------|
| model | `3b_fp8` | Fits 16GB with swap. 7B is possible but slow — not for a first look. |
| `BATCH_SIZE` | `5` | Frames/batch (4n+1). Higher = better temporal consistency + more VRAM. Drop to `1` if OOM. |
| `BLOCKS_TO_SWAP` | `16` | Transformer blocks offloaded to CPU RAM. Raise to 24–32 if OOM; lower for speed. |
| VAE tiling | on | Needed for higher resolutions on 16GB. |
| `RESOLUTION` | arg | Target short-edge in px. |

OOM order of attack: raise `BLOCKS_TO_SWAP` → drop `BATCH_SIZE` → lower `RESOLUTION`.

## Cleanup

Everything this kit creates is under `~/ai-upscale/seedvr2/` (+ `models/SEEDVR2/`). Delete
those to remove the venv, upstream repo, weights, and eval outputs. The main project is
untouched. To retire the experiment entirely, delete the `prototype/seedvr2/` directory.
