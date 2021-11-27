import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import ParityDataset
from utils import dataloader_accuracy
import argparse


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--n_elems',
                    type=int,
                    default=15,
                    help='length of the bitstring.')
PARSER.add_argument('--n_train_elems',
                    type=int,
                    default=10,
                    help='length of the bitstring used for training.')
PARSER.add_argument('--n_train_samples',
                    type=int,
                    default=128000,
                    help='number of training samples.')
PARSER.add_argument('--n_eval_samples',
                    type=int,
                    default=10000,
                    help='number of evaluation samples')
PARSER.add_argument('--n_epochs',
                    type=int,
                    default=100,
                    help='Number of epochs to train.')
PARSER.add_argument('--n_layers',
                    type=int,
                    default=1,
                    help='Number of layers.')
PARSER.add_argument('--train_unique',
                    type=bool,
                    default='',
                    help='if the training dataset contains duplicated data.')
PARSER.add_argument('--n_exclusive_data',
                    type=int,
                    default=0,
                    help='number of data that the training data does not contain.')
PARSER.add_argument('--data_augmentation',
                    type=float,
                    default=0,
                    help='Augment the dataset by the specified ratio')
PARSER.add_argument('--log_folder',
                    type=str,
                    default='results',
                    help='log folder')


args = PARSER.parse_args()
print(args)

exclusive_data = ParityDataset(n_samples=args.n_exclusive_data,
                               n_elems=args.n_elems,
                               n_nonzero_min=1,
                               n_nonzero_max=args.n_train_elems,
                               exclude_dataset=None,
                               unique=True,
                               model='rnn')
train_data = ParityDataset(n_samples=args.n_train_samples,
                           n_elems=args.n_elems,
                           n_nonzero_min=1,
                           n_nonzero_max=args.n_train_elems,
                           exclude_dataset=exclusive_data,
                           unique=args.train_unique,
                           model='rnn',
                           data_augmentation=args.data_augmentation)
val_data = ParityDataset(n_samples=args.n_eval_samples,
                         n_elems=args.n_elems,
                         n_nonzero_min=1,
                         n_nonzero_max=args.n_train_elems,
                         exclude_dataset=train_data,
                         unique=True,
                         model='rnn')
extra_data = ParityDataset(n_samples=args.n_eval_samples if args.n_elems != args.n_train_elems else 0,
                           n_elems=args.n_elems,
                           n_nonzero_min=args.n_train_elems,
                           n_nonzero_max=args.n_elems,
                           exclude_dataset=None,
                           unique=True,
                           model='rnn')

batch_size = 128
train_dataloader = DataLoader(train_data, batch_size=batch_size)
dataloader_dict = {
    'validation': DataLoader(val_data, batch_size=batch_size),
    'extra': DataLoader(extra_data, batch_size=batch_size),
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

lstm_model = torch.load(f'{args.n_elems}.pt')
lstm_model = lstm_model.to(device)

noise_scale = [i * 0.01 for i in range(20)]
train_acc = []
val_acc = []
extra_acc = []
for scale in noise_scale:
    with torch.no_grad():
        for param in lstm_model.parameters():
            noise = torch.randn(param.size()).to(device)
            param.add_(noise * scale)

        acc = dataloader_accuracy(train_dataloader, lstm_model)
        train_acc.append(acc)
        for loader_name, loader in dataloader_dict.items():
            v_acc = dataloader_accuracy(loader, lstm_model)
            if loader_name == 'validation':
                val_acc.append(v_acc)
            elif loader_name == 'extra':
                extra_acc.append(v_acc)

plt.plot(noise_scale, train_acc)
plt.plot(noise_scale, val_acc)
plt.plot(noise_scale, extra_acc)
plt.show()

