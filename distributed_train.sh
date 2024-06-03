#!/bin/bash
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port 65100 train_focalloss_tripletloss_pplcnet.py "$@"