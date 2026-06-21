#!/bin/bash
##############################################################################
# Shared paths + helpers for the SeedVR2 prototype kit.
#
# Mirrors the main project's split so this sits ALONGSIDE the existing pipeline
# rather than fighting it:
#   - Source / scripts  → the ai-upscaler git clone (this repo, tracked)
#   - Runtime artifacts → ~/ai-upscale/ (the existing runtime tree, untracked),
#                         exactly where the main script keeps venv/models/temp.
#
# The SeedVR2 venv is kept separate from ~/ai-upscale/venv on purpose (different
# torch build), but lives under the same runtime root and shares the models/ dir.
##############################################################################

RUNTIME_DIR="$HOME/ai-upscale"            # == SCRIPT_DIR in upscale_video.sh
SEEDVR2_DIR="$RUNTIME_DIR/seedvr2"        # isolated SeedVR2 runtime root
SEEDVR2_VENV="$SEEDVR2_DIR/venv"          # separate venv (torch conflict avoidance)
SEEDVR2_REPO="$SEEDVR2_DIR/repo"          # upstream numz/ComfyUI-SeedVR2 checkout
MODEL_DIR="$RUNTIME_DIR/models/SEEDVR2"   # shares the main models/ tree
EVAL_DIR="$SEEDVR2_DIR/eval"              # clips + outputs (runtime, not in git)
CLIPS_DIR="$EVAL_DIR/clips"
OUT_DIR="$EVAL_DIR/out"
CLI="$SEEDVR2_REPO/inference_cli.py"

# Path to the main upscaler, for A/B comparison runs.
MAIN_UPSCALER="$RUNTIME_DIR/upscale_video.sh"

# Colours + helpers — same palette as upscale_video.sh / install.sh.
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }
