import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import ParityDataset
from utils import batch_accuracy, dataloader_accuracy
from models import LSTM
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
                    default=50,
                    help='Number of epochs to train.')
PARSER.add_argument('--n_layers',
                    type=int,
                    default=1,
                    help='Number of layers.')
PARSER.add_argument('--train_unique',
                    type=bool,
                    default=False,
                    help='if the training dataset contains duplicated data.')
PARSER.add_argument('--n_exclusive_data',
                    type=int,
                    default=10000,
                    help='number of data that the training data does not contain.')
PARSER.add_argument('--log_folder',
                    type=str,
                    default='results',
                    help='log folder')



args = PARSER.parse_args()
print(args)

n_elems = args.n_elems
n_train_elems = args.n_train_elems
n_train_samples = args.n_train_samples
num_epochs = args.n_epochs
n_eval_samples = args.n_eval_samples
train_unique = args.train_unique
n_exclusive_data = args.n_exclusive_data

exclusive_data = ParityDataset(n_samples=n_exclusive_data,
                               n_elems=n_elems,
                               n_nonzero_min=1,
                               n_nonzero_max=n_train_elems,
                               exclude_dataset=None,
                               unique=True,
                               model='rnn')

extra_data = ParityDataset(n_samples=n_eval_samples if n_elems != n_train_elems else 0,
                           n_elems=n_elems,
                           n_nonzero_min=n_train_elems,
                           n_nonzero_max=n_elems,
                           exclude_dataset=None,
                           unique=True,
                           model='rnn')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

input_size = 1
hidden_size = 128
num_layers = args.n_layers
learning_rate = 0.0003
eval_interval = 50
lstm_model = LSTM(input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers)
lstm_model = lstm_model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
writer = SummaryWriter(f'{args.log_folder}/lstm{n_elems}_{n_train_elems}' +
                       f'_{num_layers}_{num_epochs}_{n_eval_samples}_{n_train_samples}' +
                       f'_{train_unique}-{n_exclusive_data}')


num_steps = 0
for num_epoch in range(num_epochs):
    train_data = ParityDataset(n_samples=n_train_samples,
                               n_elems=n_elems,
                               n_nonzero_min=1,
                               n_nonzero_max=n_train_elems,
                               exclude_dataset=exclusive_data,
                               unique=train_unique,
                               model='rnn')
    val_data = ParityDataset(n_samples=n_eval_samples,
                             n_elems=n_elems,
                             n_nonzero_min=1,
                             n_nonzero_max=n_train_elems,
                             exclude_dataset=train_data,
                             unique=True,
                             model='rnn')
    batch_size = 128
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    dataloader_dict = {
        'validation': DataLoader(val_data, batch_size=batch_size),
        'extra': DataLoader(extra_data, batch_size=batch_size),
    }

    print(f'Epochs: {num_epoch}')
    for X_batch, y_batch in train_dataloader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred_batch = lstm_model(X_batch)[:, 0]
        train_batch_acc = batch_accuracy(y_pred_batch, y_batch)
        writer.add_scalar('train_batch_accuracy', train_batch_acc, num_steps)

        loss = criterion(y_pred_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (num_steps % eval_interval) == 0:
            for loader_name, loader in dataloader_dict.items():
                val_acc = dataloader_accuracy(loader, lstm_model)
                writer.add_scalar(loader_name, val_acc, num_steps)

        num_steps += 1
