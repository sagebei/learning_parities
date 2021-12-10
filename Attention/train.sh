#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=48:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -l cluster=andrena  # Ensure that the job runs on Andrena nodes
#$ -N attention9      # Name for the job (optional)

## Load the necessary modules
#module load python/3.8.5
#module load cudnn/8.1.1-cuda11
#
## Load the virtualenv containing the pytorch package
#source ~/venv/bin/activate

# run the python script
python train_self_attention.py --n_elems=20 \
                               --n_train_elems=15 \
                               --n_train_samples=128000 \
                               --n_eval_samples=10000 \
                               --n_epochs=100 \
                               --batch_size=128 \
                               --n_layers=4 \
                               --train_unique='' \
                               --mode='soft' \
                               --embed_dim=36 \
                               --n_heads=3 \
                               --linear_dim=3 \
                               --dropout=0 \
                               --log_folder='results'
