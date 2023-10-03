#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=11G   # Request RAM (7.5GB per core)
#$ -l h_rt=240:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -N value      # Name for the job (optional)
#$ -t 20-100:5

# Load the necessary modules
module load python/3.8.5

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate

# $(( 1280 * $n_bits ))

for lr in 0.001 0.0025 0.005 0.0003 0.0005; do
python main.py --n_elems=${SGE_TASK_ID} \
               --n_train_elems=${SGE_TASK_ID} \
               --n_train_samples=256000 \
               --n_eval_samples=10000 \
               --n_epochs=1000 \
               --n_layers=1 \
               --train_unique='' \
               --noise='.' \
               --log_folder='/data/scratch/acw554/parity/lstm' \
               --seed=333  \
               --lr=$lr
done;