#!/bin/bash
##############################################################################
# FlashVSR — one-time setup
#
# Clones FlashVSR-Pro (standalone CLI fork of OpenImagingLab/FlashVSR) and
# builds an ISOLATED venv under ~/ai-upscale/flashvsr — alongside the existing
# runtime and the SeedVR2 venv, touching neither. Run on the GPU VM (needs CUDA).
#
# This is the slowest of the three setups because Block-Sparse-Attention is
# compiled from source against your GPU's CUDA architecture.
##############################################################################
set -eo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SELF_DIR/_env.sh"

REPO_URL="https://github.com/LujiaJin/FlashVSR-Pro.git"
HF_REPO="JunhaoZhuang/FlashVSR-v1.1"

# FlashVSR-Pro pins torch 2.6.0 + cu124 (same build family as the SeedVR2 venv).
TORCH_SPEC=(torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0)
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

command -v git     >/dev/null || { err "git not found";    exit 1; }
command -v python3 >/dev/null || { err "python3 not found"; exit 1; }
command -v ffmpeg  >/dev/null || { err "ffmpeg not found";  exit 1; }

if ! command -v nvidia-smi >/dev/null; then
    err "nvidia-smi not found — FlashVSR needs a CUDA GPU. Are you on the GPU VM?"
    exit 1
fi

if [[ ! -d "$RUNTIME_DIR" ]]; then
    err "Runtime dir $RUNTIME_DIR not found — expected the main project installed here."
    exit 1
fi

# ── Detect the GPU's CUDA architecture ───────────────────────────────────────
# This matters: FlashVSR-Pro's docs suggest BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;90;100",
# which OMITS 89 — the arch of Ada cards including the RTX 4060 Ti. Building without
# your own arch means the kernels either fall back to slow PTX JIT or fail outright.
# We query the card and build for exactly what's present.
DETECTED_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
if [[ -z "$DETECTED_CAP" ]]; then
    warn "Could not detect compute capability — defaulting to 89 (Ada / 40-series)."
    DETECTED_CAP="89"
fi
CUDA_ARCHS="${BLOCK_SPARSE_ATTN_CUDA_ARCHS:-$DETECTED_CAP}"
info "GPU compute capability: ${DETECTED_CAP}  →  building Block-Sparse-Attention for arch ${CUDA_ARCHS}"
if [[ "${DETECTED_CAP}" -lt 80 ]] 2>/dev/null; then
    err "FlashVSR requires compute capability 8.0+ (Ampere or newer). Detected ${DETECTED_CAP}."
    exit 1
fi

mkdir -p "$FLASHVSR_DIR" "$FLASHVSR_MODEL_DIR" "$CLIPS_DIR" "$OUT_DIR"

# ── Clone / update FlashVSR-Pro (has submodules) ─────────────────────────────
if [[ -d "$FLASHVSR_REPO/.git" ]]; then
    info "Updating existing FlashVSR-Pro repo..."
    git -C "$FLASHVSR_REPO" pull --ff-only || warn "git pull failed — keeping existing checkout"
    git -C "$FLASHVSR_REPO" submodule update --init --recursive || warn "submodule update failed"
else
    info "Cloning FlashVSR-Pro → $FLASHVSR_REPO ..."
    git clone --recursive "$REPO_URL" "$FLASHVSR_REPO"
fi

# ── Isolated venv (prefer 3.11 — what upstream targets) ──────────────────────
PYBIN="python3"
for cand in python3.11 python3.12; do
    command -v "$cand" >/dev/null && { PYBIN="$cand"; break; }
done
info "Using interpreter: $PYBIN ($("$PYBIN" --version 2>&1))"

if [[ ! -d "$FLASHVSR_VENV" ]]; then
    info "Creating isolated venv at $FLASHVSR_VENV ..."
    "$PYBIN" -m venv "$FLASHVSR_VENV"
fi
# shellcheck disable=SC1091
source "$FLASHVSR_VENV/bin/activate"

info "Upgrading pip..."
pip install --upgrade pip setuptools wheel >/dev/null

info "Installing PyTorch 2.6.0 (cu124) — large download..."
pip install "${TORCH_SPEC[@]}" --index-url "$TORCH_INDEX"

info "Installing FlashVSR-Pro package + requirements..."
pip install -e "$FLASHVSR_REPO"
[[ -f "$FLASHVSR_REPO/requirements.txt" ]] && pip install -r "$FLASHVSR_REPO/requirements.txt"

# ── Compile Block-Sparse-Attention ───────────────────────────────────────────
BSA_DIR="$FLASHVSR_REPO/Block-Sparse-Attention"
if [[ -d "$BSA_DIR" ]]; then
    if python3 -c "import block_sparse_attn" &>/dev/null; then
        ok "Block-Sparse-Attention already installed"
    else
        info "Compiling Block-Sparse-Attention for arch ${CUDA_ARCHS} (SLOW — 10-40 min)..."
        pip install packaging ninja
        (
            cd "$BSA_DIR"
            export BLOCK_SPARSE_ATTN_CUDA_ARCHS="$CUDA_ARCHS"
            python setup.py install
        ) || warn "Block-Sparse-Attention build FAILED — FlashVSR may fall back to a slower path or error."
    fi
else
    warn "Block-Sparse-Attention submodule missing — run: git -C $FLASHVSR_REPO submodule update --init --recursive"
fi

# ── Model weights ────────────────────────────────────────────────────────────
# Kept in the shared models/ tree (not inside the repo) so a repo re-clone never
# re-downloads them. infer.py finds them via the FLASHVSR-Pro_MODEL_PATH env var,
# which upscale_video.sh sets on every call.
if [[ -f "$FLASHVSR_MODEL_DIR/diffusion_pytorch_model_streaming_dmd.safetensors" ]]; then
    ok "Weights already present in $FLASHVSR_MODEL_DIR"
else
    info "Downloading FlashVSR v1.1 weights → $FLASHVSR_MODEL_DIR (several GB)..."
    # Do NOT `pip install --upgrade huggingface_hub` here: diffsynth pins it to ==0.34.4, and
    # upgrading silently breaks diffsynth/transformers/tokenizers — which makes infer.py fail to
    # import. Use whatever version the requirements already pinned; only install if truly absent.
    if ! python3 -c "import huggingface_hub" &>/dev/null; then
        info "huggingface_hub not present — installing the diffsynth-compatible pin..."
        pip install "huggingface_hub==0.34.4"
    fi
    cat > "$FLASHVSR_DIR/_dl_weights.py" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
print("weights downloaded")
PY
    if python3 "$FLASHVSR_DIR/_dl_weights.py" "$HF_REPO" "$FLASHVSR_MODEL_DIR"; then
        rm -f "$FLASHVSR_DIR/_dl_weights.py"
    else
        err "Weight download failed. Manual fallback:"
        err "  git lfs install && git clone https://huggingface.co/$HF_REPO $FLASHVSR_MODEL_DIR"
        exit 1
    fi
fi

# Guard: a mismatched huggingface_hub is the most common way this install ends up broken, because
# several packages pin it and pip will happily satisfy the newest request. Check and repair.
info "Checking dependency consistency..."
if ! pip check >/dev/null 2>&1; then
    warn "pip reports dependency conflicts:"
    pip check 2>&1 | sed 's/^/    /' || true
    HUB_PIN="$(python3 -c "import importlib.metadata as m; print(m.requires('diffsynth'))" 2>/dev/null \
               | tr ',' '\n' | grep -oE 'huggingface-hub==[0-9.]+' | head -1 | cut -d= -f3)"
    if [[ -n "$HUB_PIN" ]]; then
        info "Repairing huggingface_hub to diffsynth's pin (==${HUB_PIN})..."
        pip install "huggingface_hub==${HUB_PIN}" && ok "huggingface_hub pinned to ${HUB_PIN}"
    fi
fi

# ── Quality patch ────────────────────────────────────────────────────────────
# FlashVSR-Pro hardcodes an 8-bit H.264 CRF-20 writer even at --quality 10, which
# would cap this pipeline's quality before our encoder runs. This adds an opt-in
# 10-bit x265 writer. Non-fatal: unpatched still works, just lower quality.
info "Applying HQ (10-bit) output patch to infer.py..."
if python3 "$SELF_DIR/patch_hq_output.py" "$FLASHVSR_CLI"; then
    ok "HQ output patch active"
else
    warn "HQ patch not applied — FlashVSR will write 8-bit H.264 CRF 20."
    warn "Output remains usable, but quality is meaningfully lower. See prototype/flashvsr/README.md."
fi

# ── Sanity checks ────────────────────────────────────────────────────────────
info "Verifying CUDA in the FlashVSR venv..."
python3 - <<'PY' || warn "CUDA check failed — FlashVSR would be unusably slow on CPU."
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

info "Verifying the CLI loads..."
if python3 "$FLASHVSR_CLI" --help >/dev/null 2>&1; then
    ok "infer.py responds to --help"
else
    err "infer.py --help FAILED — the install is not usable yet. Actual error:"
    # Show the real failure rather than swallowing it; the import traceback names the culprit.
    python3 "$FLASHVSR_CLI" --help 2>&1 | tail -20 | sed 's/^/    /'
    echo ""
    err "Most common cause: a package pin was overwritten (often huggingface_hub, which diffsynth"
    err "pins to ==0.34.4). Check with:  source $FLASHVSR_VENV/bin/activate && pip check"
    exit 1
fi

echo ""
ok "Setup complete. Runtime lives under $FLASHVSR_DIR (separate from the main + SeedVR2 venvs)."
echo ""
info "Smoke-test on a SHORT clip before committing to a long run:"
echo "    ~/ai-upscaler/upscale_video.sh -i /path/to/10s_clip.mkv -r 1080p --prefilter none -m flashvsr"
echo ""
info "Weights: $FLASHVSR_MODEL_DIR"
