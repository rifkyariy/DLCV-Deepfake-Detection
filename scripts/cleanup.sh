#!/bin/bash

# Clean up logs and checkpoints
# Removes old logs while keeping the most recent ones

LOG_DIR="logs/inference"
CHECKPOINT_DIR="checkpoints/inference"
KEEP_DAYS=${1:-7}  # Default: keep files from last 7 days

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup Script${NC}"
echo -e "${GREEN}Keeping files from last $KEEP_DAYS days${NC}"
echo -e "${GREEN}========================================${NC}"

# Clean old logs
if [ -d "$LOG_DIR" ]; then
    echo -e "\n${YELLOW}Cleaning logs older than $KEEP_DAYS days...${NC}"
    old_logs=$(find "$LOG_DIR" -name "*.log" -type f -mtime +$KEEP_DAYS 2>/dev/null)
    
    if [ -n "$old_logs" ]; then
        count=$(echo "$old_logs" | wc -l)
        echo -e "${YELLOW}Found $count old log files${NC}"
        echo "$old_logs" | while read file; do
            echo "  Removing: $(basename $file)"
            rm -f "$file"
        done
        echo -e "${GREEN}✓ Cleaned $count log files${NC}"
    else
        echo -e "${GREEN}No old logs to clean${NC}"
    fi
fi

# Clean old checkpoints
if [ -d "$CHECKPOINT_DIR" ]; then
    echo -e "\n${YELLOW}Cleaning old checkpoints...${NC}"
    old_checkpoints=$(find "$CHECKPOINT_DIR" -name "*.txt" -type f -mtime +$KEEP_DAYS 2>/dev/null)
    
    if [ -n "$old_checkpoints" ]; then
        count=$(echo "$old_checkpoints" | wc -l)
        echo -e "${YELLOW}Found $count old checkpoint files${NC}"
        echo "$old_checkpoints" | while read file; do
            echo "  Removing: $(basename $file)"
            rm -f "$file"
        done
        echo -e "${GREEN}✓ Cleaned $count checkpoint files${NC}"
    else
        echo -e "${GREEN}No old checkpoints to clean${NC}"
    fi
fi

# Show current disk usage
echo -e "\n${YELLOW}Current disk usage:${NC}"
if [ -d "$LOG_DIR" ]; then
    du -sh "$LOG_DIR" 2>/dev/null | awk '{print "  Logs: " $1}'
fi
if [ -d "$CHECKPOINT_DIR" ]; then
    du -sh "$CHECKPOINT_DIR" 2>/dev/null | awk '{print "  Checkpoints: " $1}'
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup complete${NC}"
echo -e "${GREEN}========================================${NC}"
