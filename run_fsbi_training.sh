#!/bin/bash
# FSBI Training Watchdog - FIXED (Removed incompatible allocator setting)
# Features: Auto-restart on Hang/OOM, Explicit W&B Login, Live Log Streaming

set -e

# ================= CONFIGURATION =================
CONTAINER_NAME="fsbi_training"
IMAGE_NAME="sbi-transfer-learning"
HOST_PROJECT_PATH="/home/ari/SBI"
CONTAINER_APP_PATH="/app/"

# Training Configuration
GPU_ID=0
CONFIG="fsbi_src/configs/sbi/base.json"
SESSION_NAME="fsbi"
MODE="reflect"
WAVELET="sym2"

# W&B Configuration
WANDB_PROJECT="FSBI-Deepfake-Detection"
WANDB_NOTES="Training with auto-restart watchdog"

# Watchdog Configuration
MAX_RETRIES=20
TIMEOUT_SECONDS=300
CHECK_INTERVAL=30
LOG_FILE="$HOST_PROJECT_PATH/training_docker_watchdog.log"

# ================= FUNCTIONS =================

log_msg() {
    # Log to file, but only print system messages to console with a prefix
    # to distinguish them from the training logs
    echo -e "\n\033[1;33m[WATCHDOG $(date '+%H:%M:%S')] $1\033[0m" | tee -a "$LOG_FILE"
}

# W&B API Key Handling
if [ -f "$HOST_PROJECT_PATH/.env" ]; then
    set -a 
    source "$HOST_PROJECT_PATH/.env"
    set +a
fi

setup_container() {
    log_msg "Setting up Docker container..."
    
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
    fi
    
    # 1. Start Container
    docker run \
        --gpus "device=$GPU_ID" \
        --detach \
        --name $CONTAINER_NAME \
        --shm-size 64G \
        -v "$HOST_PROJECT_PATH:$CONTAINER_APP_PATH" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e WANDB_PROJECT="$WANDB_PROJECT" \
        -e WANDB_NAME="${SESSION_NAME}_${MODE}_${WAVELET}" \
        -e WANDB_NOTES="$WANDB_NOTES" \
        -e WANDB_RESUME="allow" \
        -e PYTHONUNBUFFERED=1 \
        "$IMAGE_NAME" \
        tail -f /dev/null
    
    # 2. Install Dependencies
    log_msg "Installing Python dependencies..."
    docker exec $CONTAINER_NAME bash -c '
        pip install --no-cache-dir "numpy==1.24.3" PyWavelets wandb timm 2>/dev/null || true
        pip install --no-cache-dir --force-reinstall "numpy==1.24.3" 2>/dev/null || true
    '

    # 3. EXPLICIT WANDB LOGIN
    log_msg "Authenticating W&B..."
    docker exec $CONTAINER_NAME bash -c "wandb login $WANDB_API_KEY --relogin"
    
    log_msg "Container setup complete."
}

restart_container() {
    log_msg "Restarting Docker container..."
    docker exec $CONTAINER_NAME bash -c 'pkill -9 python3 2>/dev/null || true'
    sleep 2
    docker stop $CONTAINER_NAME 2>/dev/null || true
    sleep 3
    docker start $CONTAINER_NAME 2>/dev/null || setup_container
    sleep 5
    log_msg "Container restarted."
}

check_gpu_active() {
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null || echo "0")
    GPU_UTIL=${GPU_UTIL// /}
    if [ "$GPU_UTIL" -gt 5 ]; then return 0; else return 1; fi
}

check_training_alive() {
    PROC_COUNT=$(docker exec $CONTAINER_NAME pgrep -f "train_fsbi.py" 2>/dev/null | wc -l || echo "0")
    if [ "$PROC_COUNT" -gt 0 ]; then return 0; else return 1; fi
}

run_training() {
    local attempt=$1
    log_msg "Starting training attempt $attempt/$MAX_RETRIES"
    
    RESUME_FLAG=""
    if docker exec $CONTAINER_NAME bash -c "ls output/${SESSION_NAME}_*/weights/*.tar 2>/dev/null | head -1" | grep -q ".tar"; then
        log_msg "Found existing checkpoint, will resume training"
        RESUME_FLAG="--resume"
    fi
    
    # Start training in background inside container
    # PYTHONUNBUFFERED=1 is crucial for real-time logs
    # REMOVED: expandable_segments:True (Caused the crash)
    # KEPT: max_split_size_mb:128 (Safe OOM protection)
    docker exec -d $CONTAINER_NAME bash -c "
        cd $CONTAINER_APP_PATH
        export CUDA_VISIBLE_DEVICES=0
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
        export PYTHONUNBUFFERED=1
        
        python3 fsbi_src/train_fsbi.py \
            $CONFIG \
            -n $SESSION_NAME \
            -m $MODE \
            -w $WAVELET \
            $RESUME_FLAG \
            > training_output.log 2>&1
    "
    
    log_msg "Process started. Streaming logs below..."
    log_msg "----------------------------------------"
    
    sleep 5

    # === LIVE LOG STREAMING ===
    # We stream the log file to stdout in the background
    docker exec $CONTAINER_NAME tail -f -n 20 training_output.log &
    TAIL_PID=$!
    
    # Ensure we kill the tail process if this function exits
    trap "kill $TAIL_PID 2>/dev/null" RETURN

    LAST_GPU_ACTIVITY=$(date +%s)
    
    while true; do
        sleep $CHECK_INTERVAL
        
        # Check if training is alive
        if ! check_training_alive; then
            # Analyze exit status
            EXIT_MSG=$(docker exec $CONTAINER_NAME tail -5 training_output.log 2>/dev/null || echo "")
            
            # Check for OOM specifically
            if echo "$EXIT_MSG" | grep -q "CUDA out of memory"; then
                 log_msg "ERROR: CUDA OOM detected!"
                 return 2 # Treat OOM as a restartable crash
            fi

            if echo "$EXIT_MSG" | grep -q "Training completed"; then
                log_msg "Training completed successfully!"
                return 0
            else
                log_msg "Training process died unexpectedly."
                return 1
            fi
        fi
        
        # Check GPU
        CURRENT_TIME=$(date +%s)
        if check_gpu_active; then
            LAST_GPU_ACTIVITY=$CURRENT_TIME
        else
            IDLE_TIME=$((CURRENT_TIME - LAST_GPU_ACTIVITY))
            if [ $IDLE_TIME -gt $TIMEOUT_SECONDS ]; then
                log_msg "WARNING: GPU idle for ${IDLE_TIME}s - detected hang!"
                return 2
            fi
        fi
    done
}

# ================= EXECUTION =================
log_msg "FSBI Watchdog | GPU: $GPU_ID | W&B: $WANDB_PROJECT"

setup_container

RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    run_training $RETRY_COUNT
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        exit 0
    elif [ $RESULT -eq 2 ]; then
        log_msg "Hang or OOM detected - restarting..."
        restart_container
    else
        log_msg "Fatal error - restarting..."
        sleep 5
    fi
    
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        sleep 10
    fi
done

exit 1