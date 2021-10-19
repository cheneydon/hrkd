#!/bin/bash
NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_addr 127.0.0.25 --master_port 295025 \
train_multi_domain.py --distributed "$@"
