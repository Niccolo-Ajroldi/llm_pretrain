#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ssm2

# Job specific vars
config=config/tr_7b.yaml

# Execute python script
torchrun \
  --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
  --standalone --nnodes=1 --nproc_per_node=8 \
  train.py --config=$config
