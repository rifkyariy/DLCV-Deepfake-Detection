#!/bin/bash

# Simple Inference Runner with Retry Capability
# Quick script for testing single model/dataset combinations

# Check if running inside Docker
if [ ! -f /.dockerenv ]; then
    echo "[INFO] Not running in Docker. Starting Docker container..."
    docker run --gpus all -it --shm-size 64G \
        -v /home/mitlab/research/projects/DLCV/SBI:/app/ \
        sbi-transfer-learning bash -c "cd /app && ./scripts/run_inference_simple.sh"
    exit $?
fi

# Install numpy dependency first
echo "[INFO] Installing numpy dependency..."
pip install --force-reinstall "numpy<2"

CUDA_DEVICE=0
LOG_DIR="logs/inference"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to run with retries
run_with_retry() {
    local cmd=$1
    local name=$2
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo -e "${GREEN}[$name] Attempt $attempt/$max_attempts${NC}"
        
        if eval $cmd; then
            echo -e "${GREEN}[$name] ✓ Success${NC}"
            return 0
        else
            echo -e "${RED}[$name] ✗ Failed (attempt $attempt)${NC}"
            ((attempt++))
            [ $attempt -le $max_attempts ] && sleep 30
        fi
    done
    
    echo -e "${RED}[$name] ✗ Failed after $max_attempts attempts${NC}"
    return 1
}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Simple Compressed Model Inference${NC}"
echo -e "${GREEN}Timestamp: $TIMESTAMP${NC}"
echo -e "${GREEN}========================================${NC}"

# Check GPU
echo -e "\n${GREEN}GPU Status:${NC}"
nvidia-smi -i $CUDA_DEVICE --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv

# EfficientNetV2-M inference on CDF
echo -e "\n${GREEN}=== EfficientNetV2-M (16-bit) on CDF ===${NC}"
run_with_retry \
    "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 src/inference/EfficientNetv2-M/inference_dataset.py -w weights/efficientnetv2-m-16bit-compressed.pth -d CDF --compressed" \
    "EfficientNetV2-M-CDF" \
    | tee "$LOG_DIR/efficientnetv2-m-cdf_${TIMESTAMP}.log"

# EfficientNetV2-S inference on CDF
echo -e "\n${GREEN}=== EfficientNetV2-S (16-bit) on CDF ===${NC}"
run_with_retry \
    "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 src/inference/EfficientNetv2-S/inference_dataset.py -w weights/efficientnetv2-s-16bit-compressed.pth -d CDF --compressed" \
    "EfficientNetV2-S-CDF" \
    | tee "$LOG_DIR/efficientnetv2-s-cdf_${TIMESTAMP}.log"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All inference tasks completed${NC}"
echo -e "${GREEN}Logs saved to: $LOG_DIR${NC}"
echo -e "${GREEN}========================================${NC}"
