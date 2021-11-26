#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=72:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -l cluster=andrena  # Ensure that the job runs on Andrena nodes
#$ -N ponder15      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5
module load cudnn/7.6-cuda-10.1

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate

# run the python script
for n_train_samples in 128000 256000 512000 768000; do
python train.py --n_elems=15 \
                --n_train_elems=10 \
                --n_train_samples=$n_train_samples \
                --n_eval_samples=10000 \
                --n_epochs=500 \
                --train_unique='' \
                --n_exclusive_data=0 \
                --data_augmentation=0 \
                --log_folder='results15'
done;