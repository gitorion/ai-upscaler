#!/bin/bash
##############################################################################
# Shared paths + helpers for the FlashVSR kit.
#
# Same split as the SeedVR2 kit so both engines sit ALONGSIDE the main pipeline:
#   - Source / scripts  → the ai-upscaler git clone (this repo, tracked)
#   - Runtime artifacts → ~/ai-upscale/ (untracked runtime tree)
#
# FlashVSR gets its own venv (it needs a pinned torch 2.6.0+cu124 plus a
# CUDA-compiled Block-Sparse-Attention), kept separate from both the main venv
# and the SeedVR2 venv. Weights share the main models/ tree.
##############################################################################

RUNTIME_DIR="$HOME/ai-upscale"                     # == SCRIPT_DIR in upscale_video.sh
FLASHVSR_DIR="$RUNTIME_DIR/flashvsr"               # isolated FlashVSR runtime root
FLASHVSR_VENV="$FLASHVSR_DIR/venv"                 # separate venv
FLASHVSR_REPO="$FLASHVSR_DIR/repo"                 # LujiaJin/FlashVSR-Pro checkout
FLASHVSR_MODEL_DIR="$RUNTIME_DIR/models/FLASHVSR/FlashVSR-v1.1"
FLASHVSR_CLI="$FLASHVSR_REPO/infer.py"
EVAL_DIR="$FLASHVSR_DIR/eval"
CLIPS_DIR="$EVAL_DIR/clips"
OUT_DIR="$EVAL_DIR/out"

MAIN_UPSCALER="$RUNTIME_DIR/upscale_video.sh"

# Colours + helpers — same palette as upscale_video.sh / install.sh.
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }
