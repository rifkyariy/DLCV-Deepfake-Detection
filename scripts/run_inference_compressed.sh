#!/bin/bash

# Comprehensive Inference Runner with Auto-Restart Mechanism
# Handles both EfficientNetV2-M and EfficientNetV2-S compressed models

# Check if running inside Docker
if [ ! -f /.dockerenv ]; then
    echo "[INFO] Not running in Docker. Starting Docker container..."
    docker run --gpus all -it --shm-size 64G \
        -v /home/mitlab/research/projects/DLCV/SBI:/app/ \
        sbi-transfer-learning bash -c "cd /app && ./scripts/run_inference_compressed.sh"
    exit $?
fi

# Install numpy dependency first
echo "[INFO] Installing numpy dependency..."
pip install --force-reinstall "numpy<2"

# Configuration
CUDA_DEVICE=0
LOG_DIR="logs/inference"
CHECKPOINT_DIR="checkpoints/inference"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if GPU is responsive
check_gpu_status() {
    local gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $CUDA_DEVICE 2>/dev/null || echo "0")
    echo $gpu_usage
}

# Function to run inference with auto-restart
run_inference_with_restart() {
    local model_name=$1
    local weight_path=$2
    local dataset=$3
    local max_retries=5
    local retry_count=0
    local success=0
    
    local log_file="$LOG_DIR/${model_name}_${dataset}_${TIMESTAMP}.log"
    local checkpoint_file="$CHECKPOINT_DIR/${model_name}_${dataset}_progress.txt"
    local result_file="$LOG_DIR/${model_name}_${dataset}_result.txt"
    
    echo -e "${GREEN}===========================================================${NC}"
    echo -e "${GREEN}Starting inference: $model_name on $dataset${NC}"
    echo -e "${GREEN}Weight: $weight_path${NC}"
    echo -e "${GREEN}Log file: $log_file${NC}"
    echo -e "${GREEN}===========================================================${NC}"
    
    while [ $retry_count -lt $max_retries ] && [ $success -eq 0 ]; do
        echo -e "${YELLOW}Attempt $((retry_count + 1))/$max_retries${NC}" | tee -a "$log_file"
        
        # Record start time
        local start_time=$(date +%s)
        echo "Started at: $(date)" >> "$log_file"
        
        # Run inference with timeout monitoring
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 src/inference/$model_name/inference_dataset.py \
            -w "$weight_path" \
            -d "$dataset" \
            -n 32 \
            --compressed 2>&1 | tee -a "$log_file" &
        
        local pid=$!
        local last_gpu_check=$(date +%s)
        local no_activity_count=0
        
        # Monitor the process
        while kill -0 $pid 2>/dev/null; do
            sleep 10
            
            local current_time=$(date +%s)
            local elapsed=$((current_time - start_time))
            
            # Check GPU usage every 30 seconds
            if [ $((current_time - last_gpu_check)) -ge 30 ]; then
                local gpu_usage=$(check_gpu_status)
                echo "[$(date +%H:%M:%S)] GPU Usage: ${gpu_usage}% | Elapsed: ${elapsed}s" | tee -a "$log_file"
                
                # If GPU usage is 0 for extended period, something is wrong
                if [ "$gpu_usage" -lt 5 ]; then
                    ((no_activity_count++))
                    echo -e "${YELLOW}Warning: Low GPU activity detected ($no_activity_count/6)${NC}" | tee -a "$log_file"
                    
                    if [ $no_activity_count -ge 6 ]; then
                        echo -e "${RED}GPU not responding! Killing process...${NC}" | tee -a "$log_file"
                        kill -9 $pid 2>/dev/null
                        break
                    fi
                else
                    no_activity_count=0
                fi
                
                last_gpu_check=$current_time
            fi
        done
        
        # Check if process completed successfully
        wait $pid
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            # Check if result was actually written
            if grep -q "AUC:" "$log_file"; then
                echo -e "${GREEN}✓ Inference completed successfully!${NC}" | tee -a "$log_file"
                
                # Extract and save result
                grep "AUC:" "$log_file" | tail -1 > "$result_file"
                
                # Clean up checkpoint
                rm -f "$checkpoint_file"
                success=1
            else
                echo -e "${RED}✗ Process completed but no result found${NC}" | tee -a "$log_file"
                ((retry_count++))
            fi
        else
            echo -e "${RED}✗ Process failed with exit code: $exit_code${NC}" | tee -a "$log_file"
            ((retry_count++))
            
            # Save checkpoint
            echo "retry_count=$retry_count" > "$checkpoint_file"
            echo "last_attempt=$(date)" >> "$checkpoint_file"
            
            # Wait before retry
            if [ $retry_count -lt $max_retries ]; then
                echo -e "${YELLOW}Waiting 30 seconds before retry...${NC}" | tee -a "$log_file"
                sleep 30
            fi
        fi
    done
    
    if [ $success -eq 0 ]; then
        echo -e "${RED}✗ Failed after $max_retries attempts${NC}" | tee -a "$log_file"
        return 1
    fi
    
    return 0
}

# Main execution
echo -e "${BLUE}===========================================================${NC}"
echo -e "${BLUE}Compressed Model Inference Pipeline${NC}"
echo -e "${BLUE}Timestamp: $TIMESTAMP${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Check if GPU is available
if ! nvidia-smi &>/dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. GPU not available?${NC}"
    exit 1
fi

echo -e "${GREEN}GPU Status:${NC}"
nvidia-smi -i $CUDA_DEVICE

# Array of models and datasets to test
declare -A models=(
    ["EfficientNetv2-M"]="weights/efficientnetv2-m-16bit-compressed.pth"
    ["EfficientNetv2-S"]="weights/efficientnetv2-s-16bit-compressed.pth"
)

datasets=("CDF" "FF" "DFD" "DFDC" "DFDCP" "FFIW")

# Run inference for each model and dataset
total_tasks=$((${#models[@]} * ${#datasets[@]}))
current_task=0

for model in "${!models[@]}"; do
    weight_path="${models[$model]}"
    
    # Check if weight file exists
    if [ ! -f "$weight_path" ]; then
        echo -e "${RED}Warning: Weight file not found: $weight_path${NC}"
        echo -e "${YELLOW}Skipping $model...${NC}"
        continue
    fi
    
    for dataset in "${datasets[@]}"; do
        ((current_task++))
        echo ""
        echo -e "${BLUE}===========================================================${NC}"
        echo -e "${BLUE}Task $current_task/$total_tasks: $model on $dataset${NC}"
        echo -e "${BLUE}===========================================================${NC}"
        
        run_inference_with_restart "$model" "$weight_path" "$dataset"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to complete inference for $model on $dataset${NC}"
            # Continue with next task instead of stopping
        fi
        
        # Small delay between tasks
        sleep 5
    done
done

# Generate summary report
echo -e "\n${GREEN}===========================================================${NC}"
echo -e "${GREEN}Inference Summary${NC}"
echo -e "${GREEN}===========================================================${NC}"

summary_file="$LOG_DIR/summary_${TIMESTAMP}.txt"
echo "Inference Results Summary - $(date)" > "$summary_file"
echo "==========================================================" >> "$summary_file"

for model in "${!models[@]}"; do
    echo "" >> "$summary_file"
    echo "Model: $model" >> "$summary_file"
    echo "---" >> "$summary_file"
    
    for dataset in "${datasets[@]}"; do
        result_file="$LOG_DIR/${model}_${dataset}_result.txt"
        if [ -f "$result_file" ]; then
            cat "$result_file" >> "$summary_file"
        else
            echo "$dataset| Status: FAILED or INCOMPLETE" >> "$summary_file"
        fi
    done
done

cat "$summary_file"

echo -e "\n${GREEN}===========================================================${NC}"
echo -e "${GREEN}All tasks completed!${NC}"
echo -e "${GREEN}Summary saved to: $summary_file${NC}"
echo -e "${GREEN}Detailed logs in: $LOG_DIR/${NC}"
echo -e "${GREEN}===========================================================${NC}"
