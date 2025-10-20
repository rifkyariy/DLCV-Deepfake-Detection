#!/bin/bash
# Knowledge Distillation Training Runner
# Teacher: EfficientNet-B4 (weights/FFc23.pth)
# Student: MobileViTv2-2.0 (with Apple pretrained weights)

set -e

# Configuration
CONTAINER_NAME="sbi_container"
IMAGE_NAME="sbi-transfer-learning"
HOST_PROJECT_PATH="/home/mitlab/research/projects/DLCV/SBI"
CONTAINER_APP_PATH="/app/"

# Distillation paths
DISTILL_SCRIPT="/app/src/distillation/MobileViTv2-2.0/main_distillation.py"
DISTILL_CONFIG="/app/src/distillation/MobileViTv2-2.0/configs/distill_MobileViTv2.json"
TEACHER_WEIGHTS="/app/weights/FFc23.pth"

# Session name (can be overridden with argument)
SESSION_NAME="${1:-distillation_mobilevitv2_$(date +%m%d_%H%M)}"

echo "=========================================="
echo "Knowledge Distillation Training"
echo "=========================================="
echo "Container: $CONTAINER_NAME"
echo "Session: $SESSION_NAME"
echo "Teacher: EfficientNet-B4 (weights/FFc23.pth)"
echo "Student: MobileViTv2-2.0"
echo "Config: $DISTILL_CONFIG"
echo "=========================================="
echo ""

# Check if container exists
if [ ! "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "[ERROR] Container '$CONTAINER_NAME' not found!"
    echo "[INFO] Please run ./run.sh first to create the container."
    exit 1
fi

# Start container if stopped
echo "[INFO] Ensuring container is running..."
docker start $CONTAINER_NAME > /dev/null 2>&1 || true

# Check if teacher weights exist
echo "[INFO] Checking teacher weights..."
if ! docker exec $CONTAINER_NAME test -f "$TEACHER_WEIGHTS"; then
    echo "[ERROR] Teacher weights not found: $TEACHER_WEIGHTS"
    echo "[ERROR] Please ensure weights/FFc23.pth exists in the project directory"
    exit 1
fi

# Check if config exists
echo "[INFO] Checking distillation config..."
if ! docker exec $CONTAINER_NAME test -f "$DISTILL_CONFIG"; then
    echo "[ERROR] Config file not found: $DISTILL_CONFIG"
    exit 1
fi

# Install any missing dependencies
echo "[INFO] Checking dependencies..."
docker exec $CONTAINER_NAME bash -c '
    python3 -c "import timm" 2>/dev/null || pip install --no-cache-dir timm
    python3 -c "import sklearn" 2>/dev/null || pip install --no-cache-dir scikit-learn
    python3 -c "import efficientnet_pytorch" 2>/dev/null || pip install --no-cache-dir efficientnet_pytorch
'

echo ""
echo "[INFO] Starting distillation training..."
echo "[INFO] Press Ctrl+C to stop training"
echo ""

# Run the distillation training
docker exec -it $CONTAINER_NAME bash -c "
    cd /app/src/distillation/MobileViTv2-2.0 && \
    python main_distillation.py configs/distill_MobileViTv2.json -n '$SESSION_NAME'
"

echo ""
echo "[INFO] Training completed or interrupted"
echo "[INFO] Output saved to: src/distillation/MobileViTv2-2.0/output/$SESSION_NAME*/"
