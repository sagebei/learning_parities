#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=24:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -N policy8      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate


for lr in 0.001 0.0001 0.0003 0.0005; do
  for num_layers in 1 2 3 4 5; do
    python main.py --n_piles=8 \
                   --num_layers=$num_layers \
                   --n_train_samples=5000 \
                   --n_test_samples=1000 \
                   --n_epochs=1000 \
                   --lr=$lr
done;
done;