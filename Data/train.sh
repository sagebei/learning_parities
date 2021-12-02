#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=24:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -l cluster=andrena  # Ensure that the job runs on Andrena nodes
#$ -N attention      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5
module load cudnn/8.1.1-cuda11

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate


for noise in '' '.'; do
  for n_elems in 30 40 50; do
python train.py --n_elems=$n_elems \
                --n_train_elems=$n_elems \
                --n_train_samples=128000 \
                --n_eval_samples=10000 \
                --n_epochs=200 \
                --n_layers=1 \
                --train_unique='' \
                --n_exclusive_data=0 \
                --data_augmentation=0 \
                --noise=$noise \
                --seed=0 \
                --log_folder='results'
done;
done;

