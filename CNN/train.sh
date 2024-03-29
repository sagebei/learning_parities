#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=72:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -l cluster=andrena  # Ensure that the job runs on Andrena nodes
#$ -N cnn      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5
module load cudnn/8.1.1-cuda11

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate

for n_layer in 3 5 7 9 11 13 15 17 19 21 23 25 27 29; do
python train_cnn.py --n_elems=120 \
                    --n_train_elems=115 \
                    --n_train_samples=1024000 \
                    --n_eval_samples=10000 \
                    --n_epochs=200 \
                    --n_layers=$n_layer \
                    --n_out_channel=128 \
                    --train_unique='' \
                    --log_folder='results120'
done;
