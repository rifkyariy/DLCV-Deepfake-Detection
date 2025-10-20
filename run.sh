#!/bin/bash

# Configuration
CONTAINER_NAME="sbi_container"
IMAGE_NAME="sbi-transfer-learning"
HOST_PROJECT_PATH="/home/mitlab/research/projects/DLCV/SBI"
CONTAINER_APP_PATH="/app/"

# W&B API Key Handling
if [ -f .env ]; then
  echo "[INFO] Loading WANDB_API_KEY from .env file..."
  set -a 
  source .env
  set +a
fi

if [ -z "$WANDB_API_KEY" ]; then
  echo "[PROMPT] W&B API Key not found. Please enter it."
  read -sp "Enter your Weights & Biases API key: " WANDB_API_KEY
  echo
  if [ -z "$WANDB_API_KEY" ]; then
    echo "[ERROR] No API key provided. Exiting."
    exit 1
  fi
fi

# Main Docker Logic
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "[INFO] Found existing container '$CONTAINER_NAME'. Ensuring it is running."
    docker start $CONTAINER_NAME > /dev/null
    
    # Check if timm is already installed
    echo "[INFO] Checking dependencies..."
    TIMM_INSTALLED=$(docker exec $CONTAINER_NAME python3 -c "import timm; print('installed')" 2>/dev/null || echo "missing")
    
    if [ "$TIMM_INSTALLED" = "missing" ]; then
        echo "[INFO] Installing missing dependencies..."
        docker exec $CONTAINER_NAME bash -c 'pip install --no-cache-dir "numpy<2" wandb timm'
    else
        echo "[INFO] All dependencies already installed."
    fi
else
    # Create new container from scratch
    echo "[INFO] Starting a new container named '$CONTAINER_NAME'..."
    docker run \
        --gpus all \
        --detach \
        --name $CONTAINER_NAME \
        --shm-size 64G \
        -v "$HOST_PROJECT_PATH:$CONTAINER_APP_PATH" \
        "$IMAGE_NAME" \
        tail -f /dev/null

    echo "[INFO] Container started. Running initial setup..."
    
    # Install dependencies with proper error handling
    echo "[INFO] Installing Python dependencies..."
    docker exec $CONTAINER_NAME bash -c '
        set -e
        echo "Installing numpy<2..."
        pip install --no-cache-dir "numpy<2"
        
        echo "Installing wandb..."
        pip install --no-cache-dir wandb
        
        echo "Installing timm..."
        pip install --no-cache-dir timm
        
        echo "All dependencies installed successfully."
    '
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies. Exiting."
        exit 1
    fi
    
    # Login to W&B
    echo "[INFO] Logging in to Weights & Biases..."
    docker exec -e WANDB_API_KEY=$WANDB_API_KEY $CONTAINER_NAME wandb login
fi

# Check if train_monitor.sh exists
MONITOR_SCRIPT="./train_monitor.sh"
if [ ! -f "$MONITOR_SCRIPT" ]; then
    echo "[ERROR] Monitor script not found: $MONITOR_SCRIPT"
    echo "[ERROR] Current directory: $(pwd)"
    echo "[ERROR] Files available:"
    ls -la *.sh 2>/dev/null || echo "No .sh files found"
    exit 1
fi

echo "[INFO] Copying monitor script into the container..."
docker cp "$MONITOR_SCRIPT" "$CONTAINER_NAME:${CONTAINER_APP_PATH}train_monitor.sh"

# Verify the copy was successful
if ! docker exec $CONTAINER_NAME test -f "${CONTAINER_APP_PATH}train_monitor.sh"; then
    echo "[ERROR] Failed to copy monitor script to container"
    exit 1
fi

# Make it executable
docker exec $CONTAINER_NAME chmod +x "${CONTAINER_APP_PATH}train_monitor.sh"

echo ""
echo "=========================================="
echo "Starting Automated Training Monitor"
echo "=========================================="
echo "Container: $CONTAINER_NAME"
echo "Project Path: $HOST_PROJECT_PATH"
echo "Monitor Script: ${CONTAINER_APP_PATH}train_monitor.sh"
echo ""
echo "Controls:"
echo "  - Detach: Ctrl+P, Ctrl+Q (training continues)"
echo "  - Stop: Ctrl+C (kills monitor)"
echo ""
echo "Useful Commands:"
echo "  - View logs: docker logs -f $CONTAINER_NAME"
echo "  - Shell access: docker exec -it $CONTAINER_NAME bash"
echo "  - Stop container: docker stop $CONTAINER_NAME"
echo "=========================================="
echo ""

# Execute the monitor script inside the container
echo "[INFO] Executing monitor script..."
docker exec -it $CONTAINER_NAME bash -c "cd ${CONTAINER_APP_PATH} && ./train_monitor.sh"