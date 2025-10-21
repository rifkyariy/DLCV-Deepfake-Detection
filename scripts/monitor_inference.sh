#!/bin/bash

#!/bin/bash
# Real-time Inference Monitor
# Displays GPU stats, process info, and log output

set -e

# Note: This monitoring script is designed to run on the HOST machine
# It monitors processes running inside Docker containers
echo "[INFO] Running inference monitor on host machine"
echo "[INFO] Monitoring Docker containers and GPU usage..."

# Configuration
REFRESH_INTERVAL=5
LOG_DIR="logs/inference"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

while true; do
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}   Inference Progress Monitor${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Time: ${GREEN}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
    
    # GPU Status
    echo -e "${YELLOW}=== GPU Status ===${NC}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
        awk -F', ' '{printf "GPU %s: %s\n  Usage: %s | Memory: %s/%s (%s) | Temp: %s\n", $1, $2, $3, $5, $6, $4, $7}'
    echo ""
    
    # Running Python Processes
    echo -e "${YELLOW}=== Active Inference Processes ===${NC}"
    inference_procs=$(ps aux | grep "python.*inference_dataset.py" | grep -v grep)
    if [ -z "$inference_procs" ]; then
        echo -e "${RED}No active inference processes${NC}"
    else
        echo "$inference_procs" | awk '{printf "PID: %s | CPU: %s%% | MEM: %s%% | %s\n", $2, $3, $4, $11}'
    fi
    echo ""
    
    # Recent Log Activity
    echo -e "${YELLOW}=== Recent Log Activity ===${NC}"
    if [ -d "logs/inference" ]; then
        latest_log=$(ls -t logs/inference/*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            echo -e "${GREEN}Latest log: $(basename $latest_log)${NC}"
            tail -n 8 "$latest_log" | sed 's/^/  /'
        else
            echo -e "${RED}No log files found${NC}"
        fi
    else
        echo -e "${RED}Log directory not found${NC}"
    fi
    echo ""
    
    # Checkpoint Status
    echo -e "${YELLOW}=== Checkpoint Status ===${NC}"
    if [ -d "checkpoints/inference" ]; then
        checkpoint_count=$(ls checkpoints/inference/*.txt 2>/dev/null | wc -l)
        if [ $checkpoint_count -gt 0 ]; then
            echo -e "${YELLOW}Active checkpoints: $checkpoint_count${NC}"
            ls -lh checkpoints/inference/*.txt 2>/dev/null | tail -3 | awk '{print "  " $9 " (" $5 ")"}'
        else
            echo -e "${GREEN}No active checkpoints (all tasks completed or not started)${NC}"
        fi
    else
        echo -e "${GREEN}No checkpoint directory${NC}"
    fi
    echo ""
    
    # Results Summary
    echo -e "${YELLOW}=== Completed Results ===${NC}"
    if [ -d "logs/inference" ]; then
        result_files=$(find logs/inference -name "*_result.txt" 2>/dev/null)
        if [ -n "$result_files" ]; then
            result_count=$(echo "$result_files" | wc -l)
            echo -e "${GREEN}Completed: $result_count tasks${NC}"
            echo "$result_files" | tail -5 | while read file; do
                echo -n "  $(basename $file .txt): "
                cat "$file"
            done
        else
            echo -e "${RED}No results yet${NC}"
        fi
    fi
    echo ""
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "Refreshing every ${REFRESH_INTERVAL}s | Press ${RED}Ctrl+C${NC} to exit"
    
    sleep $REFRESH_INTERVAL
done
