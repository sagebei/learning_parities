#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 1        # Request cores (8 per GPU)
#$ -l h_vmem=16G   # Request RAM (11GB per core)
#$ -l h_rt=240:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -t 6-30
#$ -N policy      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate


for num_layers in 1 2 3; do
for lr in 0.001 0.0025 0.003 0.004 0.005 0.0002 0.0003 0.0005 0.0001; do
    python main.py --n_piles=${SGE_TASK_ID} \
                   --num_layers=$num_layers \
                   --lr=$lr
done;
done;