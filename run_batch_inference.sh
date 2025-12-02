#!/bin/bash
# Batch inference script for FSBI detector
# Processes all images in the specified input directory

set -e

# Configuration
INPUT_DIR="${1:-images}"
WEIGHTS="${2:-weights/best_model.tar}"
OUTPUT_ROOT="${3:-output}"
DEVICE="${4:-cuda}"
CAM_TARGET="${5:-fake}"
PROCESS_NAME="fsbi_inference"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    echo "Usage: $0 <input_dir> [weights] [output_root] [device] [cam_target]"
    echo "Example: $0 images weights/best_model.tar output cuda fake"
    exit 1
fi

# Validate weights file
if [ ! -f "$WEIGHTS" ]; then
    echo "Error: Weights file '$WEIGHTS' not found"
    exit 1
fi

# Count images
IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "Error: No images found in '$INPUT_DIR'"
    echo "Supported formats: .jpg, .jpeg, .png, .bmp"
    exit 1
fi

echo "========================================"
echo "FSBI Batch Inference"
echo "========================================"
echo "Input directory: $INPUT_DIR"
echo "Weights: $WEIGHTS"
echo "Output root: $OUTPUT_ROOT"
echo "Device: $DEVICE"
echo "CAM target: $CAM_TARGET"
echo "Images found: $IMAGE_COUNT"
echo "========================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Initialize counters
SUCCESS_COUNT=0
FAIL_COUNT=0
CURRENT=0

# Store failed images
FAILED_IMAGES=()

# Process each image
while IFS= read -r -d '' IMAGE_PATH; do
    CURRENT=$((CURRENT + 1))
    IMAGE_NAME=$(basename "$IMAGE_PATH")
    
    echo "[$CURRENT/$IMAGE_COUNT] Processing: $IMAGE_NAME"
    
    # Run inference
    if python3 fsbi_src/inference_image.py \
        -i "$IMAGE_PATH" \
        -w "$WEIGHTS" \
        --output-root "$OUTPUT_ROOT" \
        --device "$DEVICE" \
        --cam-target "$CAM_TARGET" \
        --process-name "$PROCESS_NAME"; then
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "  ✓ Success"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_IMAGES+=("$IMAGE_PATH")
        echo "  ✗ Failed"
    fi
    echo ""
done < <(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) -print0)

# Summary
echo "========================================"
echo "Batch Inference Complete"
echo "========================================"
echo "Total images: $IMAGE_COUNT"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "========================================"

# List failed images if any
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "Failed images:"
    for FAILED_IMG in "${FAILED_IMAGES[@]}"; do
        echo "  - $FAILED_IMG"
    done
fi

# Find and list reports
echo ""
echo "Generated reports:"
find "$OUTPUT_ROOT" -name "report.md" -type f | sort | while read -r REPORT; do
    echo "  - $REPORT"
done

echo ""
echo "Done!"
