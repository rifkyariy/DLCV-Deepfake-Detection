#!/bin/bash

# Batch Compression Script
#!/bin/bash
# Batch Compression Script
# Compresses multiple model checkpoints automatically

set -e

# Check if running inside Docker
if [ ! -f /.dockerenv ]; then
    echo "[INFO] Not running in Docker. Starting Docker container..."
    docker run --gpus all -it --shm-size 64G \
        -v /home/mitlab/research/projects/DLCV/SBI:/app/ \
        sbi-transfer-learning bash -c "cd /app && ./scripts/batch_compress.sh $*"
    exit $?
fi

# Configuration
QUANTIZE_BITS=16  # Default: 16-bit compression

COMPRESSION_DIR="src/compression"
OUTPUT_DIR="weights/compressed"
LOG_DIR="logs/compression"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

LOG_FILE="$LOG_DIR/batch_compression_${TIMESTAMP}.log"

echo -e "${BLUE}========================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}Batch Model Compression${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}Timestamp: $TIMESTAMP${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}========================================${NC}" | tee -a "$LOG_FILE"

# Function to compress a checkpoint
compress_checkpoint() {
    local checkpoint_path=$1
    local output_name=$2
    local quantize_bits=$3
    local model_type=$4
    
    local output_path="$OUTPUT_DIR/${output_name}"
    
    echo -e "\n${GREEN}Compressing: $(basename $checkpoint_path)${NC}" | tee -a "$LOG_FILE"
    echo -e "Output: $output_path" | tee -a "$LOG_FILE"
    echo -e "Quantization: ${quantize_bits}-bit" | tee -a "$LOG_FILE"
    
    python3 "$COMPRESSION_DIR/$model_type/compress_pth.py" \
        "$checkpoint_path" \
        "$output_path" \
        --quantize-bits $quantize_bits 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Compression successful${NC}" | tee -a "$LOG_FILE"
        return 0
    else
        echo -e "${RED}✗ Compression failed${NC}" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Find all .tar checkpoints in output directory
echo -e "\n${YELLOW}Searching for checkpoints...${NC}" | tee -a "$LOG_FILE"

# EfficientNetV2-S checkpoints
effnetv2s_checkpoints=$(find output/sbi_run_efficientnetv2_s_384_* -name "*_model.tar" -type f 2>/dev/null | sort)

if [ -n "$effnetv2s_checkpoints" ]; then
    echo -e "\n${BLUE}=== EfficientNetV2-S Checkpoints ===${NC}" | tee -a "$LOG_FILE"
    
    # Get the best checkpoint (highest AUC from filename)
    best_checkpoint=$(echo "$effnetv2s_checkpoints" | tail -1)
    
    if [ -n "$best_checkpoint" ]; then
        # Extract epoch and AUC from filename
        filename=$(basename "$best_checkpoint")
        epoch=$(echo "$filename" | cut -d'_' -f1)
        auc=$(echo "$filename" | cut -d'_' -f2)
        
        echo -e "${GREEN}Best checkpoint: $filename (Epoch: $epoch, AUC: $auc)${NC}" | tee -a "$LOG_FILE"
        
        # Compress with 8-bit
        compress_checkpoint "$best_checkpoint" \
            "efficientnetv2-s-8bit-epoch${epoch}-auc${auc}.pth" \
            8 \
            "EfficientNetV2-S"
        
        # Compress with 16-bit
        compress_checkpoint "$best_checkpoint" \
            "efficientnetv2-s-16bit-epoch${epoch}-auc${auc}.pth" \
            16 \
            "EfficientNetV2-S"
    fi
fi

# EfficientNetV2-M checkpoints
effnetv2m_checkpoints=$(find output/sbi_run_efficientnetv2_m_480_* -name "*_model.tar" -type f 2>/dev/null | sort)

if [ -n "$effnetv2m_checkpoints" ]; then
    echo -e "\n${BLUE}=== EfficientNetV2-M Checkpoints ===${NC}" | tee -a "$LOG_FILE"
    
    # Get the best checkpoint
    best_checkpoint=$(echo "$effnetv2m_checkpoints" | tail -1)
    
    if [ -n "$best_checkpoint" ]; then
        filename=$(basename "$best_checkpoint")
        epoch=$(echo "$filename" | cut -d'_' -f1)
        auc=$(echo "$filename" | cut -d'_' -f2)
        
        echo -e "${GREEN}Best checkpoint: $filename (Epoch: $epoch, AUC: $auc)${NC}" | tee -a "$LOG_FILE"
        
        # Compress with 8-bit
        compress_checkpoint "$best_checkpoint" \
            "efficientnetv2-m-8bit-epoch${epoch}-auc${auc}.pth" \
            8 \
            "EfficientNetV2-M"
        
        # Compress with 16-bit
        compress_checkpoint "$best_checkpoint" \
            "efficientnetv2-m-16bit-epoch${epoch}-auc${auc}.pth" \
            16 \
            "EfficientNetV2-M"
    fi
fi

# Summary
echo -e "\n${BLUE}========================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}Compression Summary${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}========================================${NC}" | tee -a "$LOG_FILE"

if [ -d "$OUTPUT_DIR" ]; then
    compressed_files=$(ls -lh "$OUTPUT_DIR"/*.pth 2>/dev/null)
    if [ -n "$compressed_files" ]; then
        echo "$compressed_files" | awk '{printf "%-60s %10s\n", $9, $5}' | tee -a "$LOG_FILE"
    else
        echo -e "${RED}No compressed files created${NC}" | tee -a "$LOG_FILE"
    fi
fi

echo -e "\n${GREEN}Compression complete!${NC}" | tee -a "$LOG_FILE"
echo -e "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
