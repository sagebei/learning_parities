#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=24:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -N lstm      # Name for the job (optional)

# Load the necessary modules
module load python/3.8.5

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate

# $(( 1280 * $n_bits ))

for lr in 0.001 0.0003 0.0005; do
  for n_bits in 20 30 40 50 60 70 80 90 100; do
python train_lstm.py --n_elems=$n_bits \
                     --n_train_elems=$n_bits \
                     --n_train_samples=256000 \
                     --n_eval_samples=10000 \
                     --n_epochs=500 \
                     --n_layers=1 \
                     --train_unique='' \
                     --noise='.' \
                     --log_folder='/data/scratch/acw554/parity/lstm' \
                     --seed=33  \
                     --lr=$lr
done;
done;