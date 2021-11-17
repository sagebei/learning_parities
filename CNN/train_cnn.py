import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import ParityDataset
from utils import batch_accuracy, dataloader_accuracy
from models import CNN
import argparse

import numpy as np
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

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
                    default=50,
                    help='Number of epochs to train.')
PARSER.add_argument('--n_layers',
                    type=int,
                    default=2,
                    help='Number of layers.')
PARSER.add_argument('--n_out_channel',
                    type=int,
                    default=128,
                    help='Number of out channel.')
PARSER.add_argument('--train_unique',
                    type=bool,
                    default='',
                    help='if the training dataset contains duplicated data.')
PARSER.add_argument('--n_exclusive_data',
                    type=int,
                    default=0,
                    help='number of data that the training data does not contain.')
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
                               model='cnn')
train_data = ParityDataset(n_samples=args.n_train_samples,
                           n_elems=args.n_elems,
                           n_nonzero_min=1,
                           n_nonzero_max=args.n_train_elems,
                           exclude_dataset=exclusive_data,
                           unique=args.train_unique,
                           model='cnn')
val_data = ParityDataset(n_samples=args.n_eval_samples,
                         n_elems=args.n_elems,
                         n_nonzero_min=1,
                         n_nonzero_max=args.n_train_elems,
                         exclude_dataset=train_data,
                         unique=True,
                         model='cnn')
extra_data = ParityDataset(n_samples=args.n_eval_samples if args.n_elems != args.n_train_elems else 0,
                           n_elems=args.n_elems,
                           n_nonzero_min=args.n_train_elems,
                           n_nonzero_max=args.n_elems,
                           exclude_dataset=None,
                           unique=True,
                           model='cnn')

batch_size = 128
train_dataloader = DataLoader(train_data, batch_size=batch_size)
dataloader_dict = {
    'validation': DataLoader(val_data, batch_size=batch_size),
    'extra': DataLoader(extra_data, batch_size=batch_size),
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


cnn_model = CNN(in_channel=args.n_elems,
                out_channel=args.n_out_channel,
                kernel_size=3,
                stride=1,
                n_layers=args.n_layers)
cnn_model = cnn_model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(cnn_model.parameters(),
                       lr=0.0003)
writer = SummaryWriter(f'{args.log_folder}/{args.n_elems}_{args.n_train_elems}' +
                       f'_{args.n_layers}_{args.n_epochs}_{args.n_eval_samples}_{args.n_train_samples}' +
                       f'_{args.train_unique}-{args.n_exclusive_data}')


num_steps = 0
eval_interval = 50
for num_epoch in range(args.n_epochs):
    print(f'Epochs: {num_epoch}')
    for X_batch, y_batch in train_dataloader:
        print(X_batch.shape)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred_batch = cnn_model(X_batch)[:, 0]
        train_batch_acc = batch_accuracy(y_pred_batch, y_batch)
        writer.add_scalar('train_batch_accuracy', train_batch_acc, num_steps)

        loss = criterion(y_pred_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (num_steps % eval_interval) == 0:
            for loader_name, loader in dataloader_dict.items():
                val_acc = dataloader_accuracy(loader, cnn_model)
                writer.add_scalar(loader_name, val_acc, num_steps)

        num_steps += 1
