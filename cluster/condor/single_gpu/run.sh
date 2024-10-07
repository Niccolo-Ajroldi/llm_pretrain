#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1

# Execute python script
python train.py --config=$config --job_idx=$job_idx
