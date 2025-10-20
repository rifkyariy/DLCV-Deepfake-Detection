#!/bin/sh

docker run --gpus all -it --shm-size 64G \
    -v /home/mitlab/research/projects/DLCV/SBI:/app/ \
    sbi-transfer-learning bash
