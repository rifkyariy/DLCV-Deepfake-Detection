#!/bin/bash
# Training script for FSBI model
# Usage: 
#   ./run_training.sh          # Start new training
#   ./run_training.sh --resume # Resume from latest checkpoint

set -e  # Exit on error

# Configuration
GPU_ID=0
CONFIG="fsbi_src/configs/sbi/base.json"
SESSION_NAME="fsbi"
MODE="reflect"
WAVELET="sym2"

# Parse arguments
RESUME_FLAG=""
if [ "$#" -gt 0 ] && [ "$1" = "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "[INFO] Resuming training from latest checkpoint"
else
    echo "[INFO] Starting new training session"
fi

# Run training
echo "[INFO] Starting training with:"
echo "  GPU: $GPU_ID"
echo "  Config: $CONFIG"
echo "  Session: $SESSION_NAME"
echo "  Mode: $MODE"
echo "  Wavelet: $WAVELET"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python3 fsbi_src/train_fsbi.py \
    $CONFIG \
    -n $SESSION_NAME \
    -m $MODE \
    -w $WAVELET \
    $RESUME_FLAG

echo "[INFO] Training completed"
