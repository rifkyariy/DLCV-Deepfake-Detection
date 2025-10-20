#!/bin/bash
# Training Monitor Script
# Automatically restarts training if GPU freeze is detected

set -e  # Exit on error
set -u  # Exit on undefined variable

# Setup logging
LOG_FILE="monitor_run.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitor script started" | tee -a "$LOG_FILE"

# Lock file prevents multiple instances
LOCK_FILE="/tmp/sbi_training_monitor.lock"
if [ -e "$LOCK_FILE" ]; then
    echo "[ERROR] Monitor script is already running. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

touch "$LOCK_FILE"
cleanup() {
    rm -f "$LOCK_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitor script stopped" | tee -a "$LOG_FILE"
}
trap cleanup EXIT INT TERM

# Training configuration
CONFIG_FILE="src/configs/sbi/efficientnetv2-s-384.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

echo "[INFO] Loading configuration from: $CONFIG_FILE" | tee -a "$LOG_FILE"

# Load parameters from JSON
load_config() {
    python3 -c "
import json
import sys

try:
    with open('${CONFIG_FILE}', 'r') as f:
        config = json.load(f)
    
    params = {
        'session_name': 'SESSION_NAME',
        'wandb_project': 'WANDB_PROJECT',
        'wandb_entity': 'WANDB_ENTITY',
        'backbone_name': 'BACKBONE_NAME',
        'num_workers': 'NUM_WORKERS',
        'batch_size': 'BATCH_SIZE',
        'early_stop_patience': 'EARLY_STOP_PATIENCE'
    }
    
    for key, var_name in params.items():
        value = config.get(key, '')
        print(f'{var_name}=\"{value}\"')
    
except Exception as e:
    print(f'echo \"[ERROR] Config loading failed: {e}\"', file=sys.stderr)
    sys.exit(1)
"
}

eval "$(load_config)"

# Fallback defaults
SESSION_NAME="${SESSION_NAME:-sbi_run}"
WANDB_PROJECT="${WANDB_PROJECT:-DeepSocial-Detector}"
WANDB_ENTITY="${WANDB_ENTITY:-rifkyariy-x}"
BACKBONE_NAME="${BACKBONE_NAME:-MobileViTv2-2.0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"

{
    echo ""
    echo "=========================================="
    echo "Configuration from $CONFIG_FILE"
    echo "=========================================="
    echo "SESSION_NAME: $SESSION_NAME"
    echo "WANDB_PROJECT: $WANDB_PROJECT"
    echo "WANDB_ENTITY: $WANDB_ENTITY"
    echo "BACKBONE_NAME: $BACKBONE_NAME"
    echo "NUM_WORKERS: $NUM_WORKERS"
    echo "BATCH_SIZE: $BATCH_SIZE"
    echo "EARLY_STOP_PATIENCE: $EARLY_STOP_PATIENCE"
    echo "=========================================="
    echo ""
} | tee -a "$LOG_FILE"

# Monitor configuration
CHECK_INTERVAL=30
FREEZE_TOLERANCE=10  # Increased from 3 to 10 (5 minutes of no activity)
INIT_WAIT=30  # Reduced from 120 to 30 seconds
MIN_ACTIVE_BEFORE_MONITORING=3

# Main monitoring loop
while true; do
    
    # Find latest checkpoint
    LATEST_SESSION_DIR=$(ls -td output/${SESSION_NAME}_* 2>/dev/null | head -n 1 || echo "")
    LATEST_CHECKPOINT=""

    if [ -n "$LATEST_SESSION_DIR" ] && [ -d "$LATEST_SESSION_DIR" ]; then
        POTENTIAL_CHECKPOINT="$LATEST_SESSION_DIR/weights/latest_checkpoint.tar"
        if [ -f "$POTENTIAL_CHECKPOINT" ]; then
            LATEST_CHECKPOINT="$POTENTIAL_CHECKPOINT"
        fi
    fi

    # Build training command
    CMD="CUDA_VISIBLE_DEVICES=0 python3 src/training/${BACKBONE_NAME}/train_sbi.py ${CONFIG_FILE}"
    CMD="$CMD -n ${SESSION_NAME}"
    CMD="$CMD --wandb-project ${WANDB_PROJECT}"
    CMD="$CMD --wandb-entity ${WANDB_ENTITY}"
    CMD="$CMD --num-workers ${NUM_WORKERS}"
    CMD="$CMD --batch-size ${BATCH_SIZE}"
    CMD="$CMD --early-stop-patience ${EARLY_STOP_PATIENCE}"

    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "[INFO] Resuming from checkpoint: $LATEST_CHECKPOINT" | tee -a "$LOG_FILE"
        CMD="$CMD --resume \"$LATEST_CHECKPOINT\""
    else
        echo "[INFO] Starting new training run: $SESSION_NAME" | tee -a "$LOG_FILE"
    fi

    {
        echo ""
        echo "=========================================="
        echo "Starting training at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        echo "Command: $CMD"
        echo "=========================================="
        echo ""
    } | tee -a "$LOG_FILE"
    
    # Execute training - show output in real-time AND log to file
    bash -c "$CMD" 2>&1 | tee -a "$LOG_FILE" &
    TRAINING_PID=$!

    echo "[INFO] Training started with PID: $TRAINING_PID" | tee -a "$LOG_FILE"
    echo "[INFO] Waiting ${INIT_WAIT}s for initialization..." | tee -a "$LOG_FILE"
    
    # Wait with progress indicator
    for i in $(seq 1 $INIT_WAIT); do
        if ! kill -0 "$TRAINING_PID" 2>/dev/null; then
            echo "[WARN] Process died after ${i}s" | tee -a "$LOG_FILE"
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then
            echo "[INFO] Initialization progress: ${i}/${INIT_WAIT}s" | tee -a "$LOG_FILE"
        fi
        sleep 1
    done
    
    # Check if process survived initialization
    if ! kill -0 "$TRAINING_PID" 2>/dev/null; then
        echo "[WARN] Process died during initialization. Checking logs..." | tee -a "$LOG_FILE"
        tail -n 20 "$LOG_FILE" | grep -i "error\|exception\|traceback" || true
        wait "$TRAINING_PID" || true
        EXIT_CODE=$?
    else
        echo "[INFO] Initialization complete. Starting GPU monitoring..." | tee -a "$LOG_FILE"
        
        freeze_counter=0
        consecutive_active=0
        last_log_line=""
        
        # Monitor loop
        while kill -0 "$TRAINING_PID" 2>/dev/null; do
            # Query GPU usage
            GPU_MEMORY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -v pid="$TRAINING_PID" '$1 == pid {print $2; exit}' || echo "0")
            UTILIZATION=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo "0")
            
            # Check if process is still writing to logs (indication it's alive)
            current_log_line=$(tail -n 1 "$LOG_FILE" 2>/dev/null || echo "")
            log_changed=0
            if [ "$current_log_line" != "$last_log_line" ]; then
                log_changed=1
                last_log_line="$current_log_line"
            fi
            
            # Check if we're in validation phase (low GPU usage is normal)
            in_validation=0
            if echo "$current_log_line" | grep -qi "val\|validation"; then
                in_validation=1
            fi
            
            # Check if GPU is active OR logs are changing
            if [ -n "$GPU_MEMORY" ] && [ "$GPU_MEMORY" != "0" ] || [ "$UTILIZATION" -gt 0 ] || [ "$log_changed" -eq 1 ]; then
                ((consecutive_active++)) || true
                freeze_counter=0
                
                if [ "$consecutive_active" -ge "$MIN_ACTIVE_BEFORE_MONITORING" ]; then
                    if [ "$in_validation" -eq 1 ]; then
                        echo "[MONITOR] PID $TRAINING_PID | GPU Memory: ${GPU_MEMORY} MiB | Utilization: ${UTILIZATION}% | Status: Validation (low GPU usage normal)"
                    else
                        echo "[MONITOR] PID $TRAINING_PID | GPU Memory: ${GPU_MEMORY} MiB | Utilization: ${UTILIZATION}% | Status: Active"
                    fi
                else
                    echo "[MONITOR] PID $TRAINING_PID | Warming up: $consecutive_active/$MIN_ACTIVE_BEFORE_MONITORING"
                fi
            else
                # GPU inactive AND logs not changing
                if [ "$consecutive_active" -ge "$MIN_ACTIVE_BEFORE_MONITORING" ]; then
                    ((freeze_counter++)) || true
                    echo "[MONITOR] PID $TRAINING_PID | No GPU activity & logs stale | Freeze counter: $freeze_counter/$FREEZE_TOLERANCE"
                else
                    echo "[MONITOR] PID $TRAINING_PID | Waiting for GPU initialization: $consecutive_active/$MIN_ACTIVE_BEFORE_MONITORING"
                fi
            fi

            # Kill if frozen (only if logs are also not changing)
            if [ "$freeze_counter" -ge "$FREEZE_TOLERANCE" ]; then
                echo "[ALERT] True freeze detected (no GPU activity AND logs stale). Killing process $TRAINING_PID" | tee -a "$LOG_FILE"
                kill -9 "$TRAINING_PID" 2>/dev/null || true
                break
            fi
            
            sleep "$CHECK_INTERVAL"
        done
        
        wait "$TRAINING_PID" || true
        EXIT_CODE=$?
    fi

    # Check exit status
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[INFO] Training completed successfully (Exit Code: 0). Exiting monitor." | tee -a "$LOG_FILE"
        break
    elif [ "$EXIT_CODE" -eq 137 ] || [ "$EXIT_CODE" -eq 143 ]; then
        # 137 = SIGKILL (kill -9), 143 = SIGTERM (kill)
        echo "[WARN] Training was killed by signal (Exit Code: $EXIT_CODE)" | tee -a "$LOG_FILE"
        echo "[INFO] This might be freeze detection or manual intervention" | tee -a "$LOG_FILE"
    else
        echo "[WARN] Training exited with error (Exit Code: $EXIT_CODE)" | tee -a "$LOG_FILE"
        echo "[WARN] Check monitor_run.log for details" | tee -a "$LOG_FILE"
    fi

    echo "[INFO] Restarting in 5 seconds..." | tee -a "$LOG_FILE"
    sleep 5
done