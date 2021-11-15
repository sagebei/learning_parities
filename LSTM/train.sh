# run the python script
#for n_elems in 20 30 40 50 60 70 80 90 100; do
python train_lstm.py --n_elems=20 \
                     --n_train_elems=15 \
                     --n_train_samples=128000 \
                     --n_eval_samples=10000 \
                     --n_epochs=50 \
                     --n_layers=1 \
                     --train_unique='' \
                     --n_exclusive_data=0 \
                     --data_augmentation=0 \
                     --log_folder='results'
#done;