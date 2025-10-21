#!/bin/bash

# Single Model Inference Script
# Run inference for a specific model and dataset with retry capability

# Usage: ./run_single_inference.sh <model> <weight_path> <dataset>
# Example: ./run_single_inference.sh EfficientNetv2-M weights/model.pth CDF

# Check if running inside Docker
if [ ! -f /.dockerenv ]; then
    echo "[INFO] Not running in Docker. Starting Docker container..."
    docker run --gpus all -it --shm-size 64G \
        -v /home/mitlab/research/projects/DLCV/SBI:/app/ \
        sbi-transfer-learning bash -c "cd /app && ./scripts/run_single_inference.sh $*"
    exit $?
fi

# Install numpy dependency first
echo "[INFO] Installing numpy dependency..."
pip install --force-reinstall "numpy<2"

CUDA_DEVICE=${CUDA_VISIBLE_DEVICES:-0}
MAX_RETRIES=3
LOG_DIR="logs/inference"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
MODEL_NAME=${1:-"EfficientNetv2-S"}
WEIGHT_PATH=${2:-"weights/efficientnetv2-s-16bit-compressed.pth"}
DATASET=${3:-"CDF"}
N_FRAMES=${4:-32}

LOG_FILE="$LOG_DIR/${MODEL_NAME}_${DATASET}_${TIMESTAMP}.log"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Single Model Inference Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Model: ${YELLOW}$MODEL_NAME${NC}"
echo -e "Weight: ${YELLOW}$WEIGHT_PATH${NC}"
echo -e "Dataset: ${YELLOW}$DATASET${NC}"
echo -e "Frames: ${YELLOW}$N_FRAMES${NC}"
echo -e "Log: ${YELLOW}$LOG_FILE${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if weight file exists
if [ ! -f "$WEIGHT_PATH" ]; then
    echo -e "${RED}Error: Weight file not found: $WEIGHT_PATH${NC}"
    exit 1
fi

# Function to run inference
run_inference() {
    local attempt=$1
    
    echo -e "\n${YELLOW}Attempt $attempt/$MAX_RETRIES${NC}" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 src/inference/$MODEL_NAME/inference_dataset.py \
        -w "$WEIGHT_PATH" \
        -d "$DATASET" \
        -n $N_FRAMES \
        --compressed 2>&1 | tee -a "$LOG_FILE"
    
    return ${PIPESTATUS[0]}
}

# Retry loop
attempt=1
success=0

while [ $attempt -le $MAX_RETRIES ] && [ $success -eq 0 ]; do
    run_inference $attempt
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        if grep -q "AUC:" "$LOG_FILE"; then
            echo -e "\n${GREEN}✓ Inference completed successfully!${NC}" | tee -a "$LOG_FILE"
            
            # Extract result
            result=$(grep "AUC:" "$LOG_FILE" | tail -1)
            echo -e "${GREEN}Result: $result${NC}"
            
            # Save result
            echo "$result" > "$LOG_DIR/${MODEL_NAME}_${DATASET}_result.txt"
            success=1
        else
            echo -e "${RED}✗ No result found in log${NC}" | tee -a "$LOG_FILE"
            ((attempt++))
        fi
    else
        echo -e "${RED}✗ Failed with exit code: $exit_code${NC}" | tee -a "$LOG_FILE"
        ((attempt++))
        
        if [ $attempt -le $MAX_RETRIES ]; then
            echo -e "${YELLOW}Retrying in 30 seconds...${NC}"
            sleep 30
        fi
    fi
done

if [ $success -eq 0 ]; then
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}✗ Failed after $MAX_RETRIES attempts${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
else
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Inference completed successfully${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
fi
