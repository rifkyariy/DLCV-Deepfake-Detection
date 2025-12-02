#!/bin/sh

docker run --gpus all -it --shm-size 64G \
    -v /home/ari/SBI:/app/ \
    sbi-transfer-learning bash
