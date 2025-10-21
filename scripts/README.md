# SBI Inference Scripts

This directory contains various scripts for running inference, monitoring progress, and managing model compression.

**⚠️ Important:** All scripts automatically start the Docker container if not already running inside one. The Docker command used is:
```bash
docker run --gpus all -it --shm-size 64G \
    -v /home/mitlab/research/projects/DLCV/SBI:/app/ \
    sbi-transfer-learning bash
```

## 📁 Scripts Overview

### 🚀 Main Inference Scripts

#### `run_inference_compressed.sh`
Comprehensive inference runner with auto-restart mechanism for all compressed models on all datasets.

**Features:**
- Runs both EfficientNetV2-M and EfficientNetV2-S
- Tests on all datasets (CDF, FF, DFD, DFDC, DFDCP, FFIW)
- Auto-restart on GPU freeze or failure (up to 5 retries)
- GPU activity monitoring
- Progress logging
- Generates summary report

**Usage:**
```bash
./run_inference_compressed.sh
```

---

#### `run_inference_simple.sh`
Quick inference runner for testing specific model/dataset combinations.

**Features:**
- Simple retry mechanism (3 attempts)
- Fast execution
- Good for quick tests

**Usage:**
```bash
./run_inference_simple.sh
```

---

#### `run_single_inference.sh`
Run inference for a single model and dataset with custom parameters.

**Usage:**
```bash
./run_single_inference.sh <model_name> <weight_path> <dataset> [n_frames]

# Examples:
./run_single_inference.sh EfficientNetv2-S weights/model.pth CDF 32
./run_single_inference.sh EfficientNetv2-M weights/model.pth FF 64
```

**Parameters:**
- `model_name`: EfficientNetv2-S or EfficientNetv2-M
- `weight_path`: Path to .pth model file
- `dataset`: CDF, FF, DFD, DFDC, DFDCP, or FFIW
- `n_frames`: Number of frames to extract (default: 32)

---

### 📊 Monitoring & Management

#### `monitor_inference.sh`
Real-time monitoring dashboard for inference progress.

**Features:**
- GPU usage statistics
- Active process information
- Recent log tail
- Checkpoint status
- Completed results summary
- Auto-refresh every 10 seconds

**Usage:**
```bash
./monitor_inference.sh
```

Press `Ctrl+C` to exit.

---

#### `cleanup.sh`
Clean up old logs and checkpoint files.

**Usage:**
```bash
# Keep files from last 7 days (default)
./cleanup.sh

# Keep files from last 3 days
./cleanup.sh 3

# Keep files from last 30 days
./cleanup.sh 30
```

---

### 🗜️ Compression Scripts

#### `batch_compress.sh`
Batch compress multiple .tar checkpoints to .pth format.

**Features:**
- Finds best checkpoints automatically
- Creates both 8-bit and 16-bit versions
- Logs all compression results
- Supports both EfficientNetV2-S and EfficientNetV2-M

**Usage:**
```bash
./batch_compress.sh
```

**Output:** Compressed models saved to `weights/compressed/`

---

## 🎯 Quick Start Guide

### Step 1: Run Simple Test
```bash
cd /home/mitlab/research/projects/DLCV/SBI
./scripts/run_inference_simple.sh
```

The script will automatically:
- Check if running in Docker
- Start Docker container if needed
- Execute inference inside the container

### Step 2: Monitor Progress (Optional)
Open a **separate terminal on the host machine** and run:
```bash
./scripts/monitor_inference.sh
```

---

## 🐳 Docker Integration

All inference and compression scripts automatically handle Docker execution:

1. **Outside Docker:** Script detects and starts Docker container
2. **Inside Docker:** Script runs directly without re-launching

### Manual Docker Usage
If you prefer to work inside Docker manually:
```bash
# Start Docker container
docker run --gpus all -it --shm-size 64G \
    -v /home/mitlab/research/projects/DLCV/SBI:/app/ \
    sbi-transfer-learning bash

# Inside container
cd /app
./scripts/run_inference_simple.sh  # Runs directly, skips Docker check
```

---

## ❓ Troubleshooting

### Q: Scripts fail with permission denied?
**A:** Make them executable:
```bash
chmod +x scripts/*.sh
```

### Q: Docker not found error?
**A:** Ensure Docker is installed and the service is running:
```bash
sudo systemctl start docker
docker ps  # Verify Docker is working
```

### Q: Docker image 'sbi-transfer-learning' not found?
**A:** Build the Docker image first:
```bash
cd /home/mitlab/research/projects/DLCV/SBI
docker build -t sbi-transfer-learning -f dockerfiles/Dockerfile .
```

### Q: Can I run scripts manually inside Docker?
**A:** Yes, start Docker manually and run scripts:
```bash
docker run --gpus all -it --shm-size 64G \
    -v /home/mitlab/research/projects/DLCV/SBI:/app/ \
    sbi-transfer-learning bash

# Inside Docker:
cd /app
./scripts/run_inference_simple.sh  # Will skip Docker check
```

### 2. Run Full Inference Pipeline
```bash
# In terminal 1: Run inference
cd /home/mitlab/research/projects/DLCV/SBI
./scripts/run_inference_compressed.sh

# In terminal 2: Monitor progress
./scripts/monitor_inference.sh
```

### 3. Test Single Model
```bash
./scripts/run_single_inference.sh EfficientNetv2-S weights/efficientnetv2-s-16bit-compressed.pth CDF
```

### 4. Compress New Models
```bash
./scripts/batch_compress.sh
```

### 5. Clean Up Old Files
```bash
./scripts/cleanup.sh 7
```

---

## 📂 Directory Structure

After running the scripts, you'll have:

```
SBI/
├── logs/
│   ├── inference/
│   │   ├── EfficientNetv2-S_CDF_20251021_123456.log
│   │   ├── EfficientNetv2-M_FF_20251021_123456.log
│   │   ├── summary_20251021_123456.txt
│   │   └── *_result.txt
│   └── compression/
│       └── batch_compression_20251021_123456.log
├── checkpoints/
│   └── inference/
│       └── *_progress.txt
└── weights/
    └── compressed/
        ├── efficientnetv2-s-8bit-epoch19-auc0.9993.pth
        └── efficientnetv2-m-16bit-epoch15-auc0.9850.pth
```

---

## 🔧 Configuration

### Environment Variables

You can customize behavior with environment variables:

```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Custom log directory
export LOG_DIR="custom_logs"

# Number of frames
export N_FRAMES=64
```

### Script Customization

Edit the script files to modify:
- `CUDA_DEVICE`: GPU device number (default: 0)
- `MAX_RETRIES`: Number of retry attempts (default: 3-5)
- `REFRESH_INTERVAL`: Monitor refresh rate (default: 10s)
- `KEEP_DAYS`: Cleanup retention days (default: 7)

---

## 🐛 Troubleshooting

### GPU Not Responding
The scripts automatically detect and restart on GPU freeze. Check:
```bash
nvidia-smi
```

### Process Stuck
Kill stuck processes:
```bash
pkill -9 -f "inference_dataset.py"
```

### Logs Too Large
Clean up old logs:
```bash
./scripts/cleanup.sh 1
```

### Out of Memory
Reduce number of frames:
```bash
./scripts/run_single_inference.sh EfficientNetv2-S weights/model.pth CDF 16
```

---

## 📈 Performance Tips

1. **Use monitoring script** to track GPU usage
2. **Run batch jobs overnight** with auto-restart enabled
3. **Clean logs regularly** to save disk space
4. **Compress models** to reduce inference time
5. **Use 16-bit quantization** for best balance of size/accuracy

---

## 🆘 Getting Help

Check the logs for detailed error messages:
```bash
# View latest log
tail -f logs/inference/*.log

# Search for errors
grep -i "error\|failed" logs/inference/*.log

# View summary
cat logs/inference/summary_*.txt
```

---

## ✅ Checklist

Before running inference:
- [ ] Models are compressed and in `weights/` directory
- [ ] GPU is available (`nvidia-smi`)
- [ ] Python environment is activated
- [ ] Datasets are properly configured
- [ ] Scripts have execute permissions

---

**Last Updated:** October 21, 2025
