#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llm

export HOME=/home/najroldi

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1

# Execute python script
torchrun \
  --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
  --standalone --nnodes=1 --nproc_per_node=8 \
  train.py --config=$config --job_idx=$job_idx
