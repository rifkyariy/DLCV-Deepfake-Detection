#!/bin/bash
# Watchdog training script - auto-restarts if training hangs
# Usage: ./run_training_watchdog.sh

set -e

# Configuration
GPU_ID=0
CONFIG="fsbi_src/configs/sbi/base.json"
SESSION_NAME="fsbi"
MODE="reflect"
WAVELET="sym2"
MAX_RETRIES=10
TIMEOUT_SECONDS=600  # 10 minutes timeout per epoch (adjust as needed)

RETRY_COUNT=0
LOG_FILE="training_watchdog.log"

echo "[$(date)] Watchdog training started" | tee -a "$LOG_FILE"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo ""
    echo "=============================================="
    echo "[$(date)] Training attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"
    echo "=============================================="
    
    # Check for existing checkpoints to resume
    RESUME_FLAG=""
    CHECKPOINT_DIRS=$(ls -d output/${SESSION_NAME}_* 2>/dev/null | tail -1 || true)
    if [ -n "$CHECKPOINT_DIRS" ]; then
        CHECKPOINT_FILES=$(ls "$CHECKPOINT_DIRS/weights/"*.tar 2>/dev/null | head -1 || true)
        if [ -n "$CHECKPOINT_FILES" ]; then
            echo "[INFO] Found checkpoint, will resume training"
            RESUME_FLAG="--resume"
        fi
    fi
    
    # Set CUDA environment variables for stability
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    
    # Run training with timeout monitoring
    # The training script now has internal CUDA sync, so we monitor externally
    START_TIME=$(date +%s)
    
    python3 fsbi_src/train_fsbi.py \
        $CONFIG \
        -n $SESSION_NAME \
        -m $MODE \
        -w $WAVELET \
        $RESUME_FLAG &
    
    TRAIN_PID=$!
    echo "[INFO] Training PID: $TRAIN_PID"
    
    # Monitor the training process
    LAST_GPU_ACTIVITY=$(date +%s)
    HANG_DETECTED=false
    
    while kill -0 $TRAIN_PID 2>/dev/null; do
        sleep 30  # Check every 30 seconds
        
        # Check GPU utilization
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null || echo "0")
        GPU_UTIL=${GPU_UTIL// /}  # Remove spaces
        
        CURRENT_TIME=$(date +%s)
        
        if [ "$GPU_UTIL" -gt 5 ]; then
            # GPU is active
            LAST_GPU_ACTIVITY=$CURRENT_TIME
        else
            # GPU is idle, check timeout
            IDLE_TIME=$((CURRENT_TIME - LAST_GPU_ACTIVITY))
            if [ $IDLE_TIME -gt $TIMEOUT_SECONDS ]; then
                echo "[$(date)] WARNING: GPU idle for ${IDLE_TIME}s (timeout: ${TIMEOUT_SECONDS}s)" | tee -a "$LOG_FILE"
                echo "[$(date)] Killing hung training process..." | tee -a "$LOG_FILE"
                kill -9 $TRAIN_PID 2>/dev/null || true
                HANG_DETECTED=true
                break
            elif [ $IDLE_TIME -gt 60 ]; then
                echo "[$(date)] GPU idle for ${IDLE_TIME}s..." 
            fi
        fi
    done
    
    # Wait for process to finish
    wait $TRAIN_PID 2>/dev/null
    EXIT_CODE=$?
    
    if [ "$HANG_DETECTED" = true ]; then
        echo "[$(date)] Training hung - will retry with resume" | tee -a "$LOG_FILE"
        # Clear GPU memory
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 5
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully!" | tee -a "$LOG_FILE"
        exit 0
    else
        echo "[$(date)] Training exited with code $EXIT_CODE" | tee -a "$LOG_FILE"
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "[$(date)] Waiting 10 seconds before retry..." | tee -a "$LOG_FILE"
        sleep 10
    fi
done

echo "[$(date)] Max retries ($MAX_RETRIES) reached. Training failed." | tee -a "$LOG_FILE"
exit 1
