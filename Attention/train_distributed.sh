#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 16        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=24:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=2         # Request GPU
#$ -N dist_attn      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5
module load cudnn/8.1.1-cuda11

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate

# run the python script
for n_layers in 1 2 3 4 5 6 7 8 9 10 11 12 13 15 16; do
python train_self_attention_distributed.py --n_elems=60 \
                                           --n_train_elems=55 \
                                           --n_train_samples=512000 \
                                           --n_eval_samples=10000 \
                                           --n_exclusive_data=0 \
                                           --n_epochs=100 \
                                           --batch_size=128 \
                                           --n_layers=$n_layers \
                                           --train_unique='' \
                                           --mode='soft' \
                                           --embed_dim=9 \
                                           --n_heads=3 \
                                           --linear_dim=9 \
                                           --dropout=0 \
                                           --log_folder='results_dist60' \
                                           --num_workers=2
done;
