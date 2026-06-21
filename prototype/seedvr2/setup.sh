#!/bin/bash
##############################################################################
# SeedVR2 prototype — one-time setup
# Clones the standalone SeedVR2 CLI and builds an ISOLATED venv under
# ~/ai-upscale/seedvr2 — alongside the existing runtime, NOT touching
# ~/ai-upscale/venv. Run on the GPU VM (needs CUDA).
##############################################################################
set -eo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SELF_DIR/_env.sh"

REPO_URL="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"

# torch build: stable cu124 works on Ada (sm_89 / 4060 Ti). The upstream README
# suggests newer/nightly builds — switch this line if requirements.txt demands it.
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

command -v git     >/dev/null || { err "git not found";     exit 1; }
command -v python3 >/dev/null || { err "python3 not found";  exit 1; }
if ! command -v nvidia-smi >/dev/null; then
    warn "nvidia-smi not found — SeedVR2 needs a CUDA GPU. Are you on the GPU VM?"
fi

if [[ ! -d "$RUNTIME_DIR" ]]; then
    err "Runtime dir $RUNTIME_DIR not found — expected the main project to be installed here."
    err "This kit is meant to run on the target machine alongside the existing pipeline."
    exit 1
fi

mkdir -p "$SEEDVR2_DIR" "$MODEL_DIR" "$CLIPS_DIR" "$OUT_DIR"

# ── Clone / update the SeedVR2 repo ──────────────────────────────────────────
if [[ -d "$SEEDVR2_REPO/.git" ]]; then
    info "Updating existing SeedVR2 repo..."
    git -C "$SEEDVR2_REPO" pull --ff-only || warn "git pull failed — keeping existing checkout"
else
    info "Cloning SeedVR2 CLI → $SEEDVR2_REPO ..."
    git clone "$REPO_URL" "$SEEDVR2_REPO"
fi

# ── Isolated venv ────────────────────────────────────────────────────────────
if [[ ! -d "$SEEDVR2_VENV" ]]; then
    info "Creating isolated venv at $SEEDVR2_VENV ..."
    python3 -m venv "$SEEDVR2_VENV"
fi
# shellcheck disable=SC1091
source "$SEEDVR2_VENV/bin/activate"

info "Upgrading pip..."
pip install --upgrade pip setuptools wheel >/dev/null

info "Installing PyTorch (CUDA) — large download..."
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

if [[ -f "$SEEDVR2_REPO/requirements.txt" ]]; then
    info "Installing SeedVR2 requirements..."
    pip install -r "$SEEDVR2_REPO/requirements.txt"
else
    warn "No requirements.txt in repo — check the SeedVR2 README for deps."
fi

# ── Sanity check ─────────────────────────────────────────────────────────────
info "Verifying CUDA in the SeedVR2 venv..."
python3 - <<'PY' || warn "CUDA check failed — SeedVR2 would be unusably slow on CPU."
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo ""
ok "Setup complete. Runtime lives under $SEEDVR2_DIR (separate from the main venv)."
echo ""
warn "Confirm the CLI flags before running run.sh:"
echo "    source $SEEDVR2_VENV/bin/activate"
echo "    python $CLI -h"
echo ""
info "Weights auto-download to $MODEL_DIR on first run (~several GB)."
info "Next: ./make_clips.sh <source>  then  ./run.sh <clip> 1080"
