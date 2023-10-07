#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 1        # Request cores (8 per GPU)
#$ -l h_vmem=16G   # Request RAM (11GB per core)
#$ -l h_rt=240:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -t 1-100
#$ -N policy      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate


python main.py --n_piles=$1 \
               --num_layers=$2 \
               --lr=${SGE_TASK_ID}
