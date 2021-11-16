#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=48:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -N mlp      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5
module load cudnn/7.6-cuda-10.1

# Load the virtualenv containing the pytorch package
source ~/project/venv/bin/activate

for n_layer in 1 2 3 4 5 6 7 8 9; do
python train_mlp.py --n_elems=30 \
                    --n_train_elems=25 \
                    --n_train_samples=256000 \
                    --n_eval_samples=10000 \
                    --n_epochs=100 \
                    --n_layers=$n_layer \
                    --train_unique='' \
                    --n_exclusive_data=0 \
                    --log_folder='results30'
done;
