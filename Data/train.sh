#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=48:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -l cluster=andrena  # Ensure that the job runs on Andrena nodes
#$ -N data      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5
module load cudnn/8.1.1-cuda11

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate

for data_augmentation in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
python train_lstm.py --n_elems=30 \
                     --n_train_elems=30 \
                     --n_train_samples=100000 \
                     --n_eval_samples=10000 \
                     --n_epochs=200 \
                     --n_layers=1 \
                     --train_unique='' \
                     --n_exclusive_data=0 \
                     --data_augmentation=$data_augmentation \
                     --log_folder='results'
done;