#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=48:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -N attention      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5
module load cudnn/7.6-cuda-10.1

# Load the virtualenv containing the pytorch package
source ~/project/venv/bin/activate

python train_mlp.py --n_elems=20 \
                    --n_train_elems=15 \
                    --n_train_samples=128000 \
                    --n_eval_samples=10000 \
                    --n_epochs=100 \
                    --n_layers=3 \
                    --train_unique='' \
                    --n_exclusive_data=0 \
                    --log_folder='results'

