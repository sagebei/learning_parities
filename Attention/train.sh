# run the python script
#
for n_layers in 1 2 3; do
python train_self_attention.py --n_elems=15 \
                               --n_train_elems=10 \
                               --n_train_samples=128000 \
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