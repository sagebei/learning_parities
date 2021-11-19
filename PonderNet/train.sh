#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=72:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -l cluster=andrena  # Ensure that the job runs on Andrena nodes
#$ -N pondernet1      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5
module load cudnn/7.6-cuda-10.1

# Load the virtualenv containing the pytorch package
source ../venv/bin/activate

# run the python script
set -x 
SEED=$RANDOM

python train.py --batch-size 128 \
                --beta 0.01 \
                --eval-frequency 4000 \
                --device cuda \
                --lambda-p 0.2 \
                --n-elems 40 \
                --n-iter 50000 \
                --n-hidden 128 \
                results/experiment_b/$SEED