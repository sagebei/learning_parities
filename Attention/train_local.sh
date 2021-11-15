# run the python script
for n_layers in 1 2 3 4 5 6 7 8 9; do
python train_self_attention.py --n_elems=30 \
                               --n_train_elems=25 \
                               --n_train_samples=256000 \
                               --n_eval_samples=10000 \
                               --n_exclusive_data=0 \
                               --n_epochs=100 \
                               --n_layers=$n_layers \
                               --train_unique='' \
                               --mode='soft' \
                               --embed_dim=30 \
                               --n_heads=3 \
                               --linear_dim=30 \
                               --dropout=0.2 \
                               --log_folder='results'
done;
