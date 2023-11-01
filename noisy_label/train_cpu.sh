#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 1        # Request cores (8 per GPU)
#$ -l h_vmem=8G   # Request RAM (7.5GB per core)
#$ -l h_rt=1:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -N value      # Name for the job (optional)
#$ -t 1-100:5

# Load the necessary modules
module load python/3.8.5

# Load the virtualenv containing the pytorch package
source ~/venv/bin/activate

python main.py --n_elems=$1 \
               --n_train_samples=12800 \
               --n_eval_samples=1000 \
               --n_epochs=1000 \
               --noisy_label=$2 \
               --n_layers=1 \
               --noise='.' \
               --seed=369  \
               --lr=${SGE_TASK_ID}