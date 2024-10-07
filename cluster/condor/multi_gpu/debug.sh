#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llm

# Job specific vars
conifg="config/first_steps/compile_no.yaml"

# Execute python script
# python -m torch.distributed.run \
torchrun \
  --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
  --standalone --nnodes=1 --nproc_per_node=8 \
  train.py --config=$conifg
