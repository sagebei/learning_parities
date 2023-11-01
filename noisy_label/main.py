import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import set_seed
from utils import ParityDataset
from utils import dataloader_accuracy, batch_accuracy
from models import LSTM
import argparse
import os, sys


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--n_elems',
                    type=int,
                    default=20,
                    help='length of the bitstring.')
PARSER.add_argument('--n_train_samples',
                    type=int,
                    default=12800,
                    help='number of training samples.')
PARSER.add_argument('--n_eval_samples',
                    type=int,
                    default=1000,
                    help='number of evaluation samples')
PARSER.add_argument('--n_epochs',
                    type=int,
                    default=1000,
                    help='Number of epochs to train.')
PARSER.add_argument('--noisy_label',
                    type=float,
                    default=0.01,
                    help='percentage of noise label.')
PARSER.add_argument('--n_layers',
                    type=int,
                    default=1,
                    help='Number of layers.')
PARSER.add_argument('--noise',
                    type=bool,
                    default='.',
                    help='if the parity data contain noise')
PARSER.add_argument('--seed',
                    type=int,
                    default=0,
                    help='seed')
PARSER.add_argument('--lr',
                    type=float,
                    default=10,
                    help='learning rate')


args = PARSER.parse_args()
print(args)

set_seed(args.seed)

result_folder = "results"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_interval = 100
lstm_model = LSTM(input_size=1,
                  hidden_size=56,
                  num_layers=args.n_layers)
lstm_model = lstm_model.to(device)

criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.0001 * args.lr
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)


num_steps = 0
train_max_acc = 0
for num_epoch in range(args.n_epochs):
    print(f'Epochs: {num_epoch}')
    noisy_label = args.noisy_label * (1 - num_epoch / args.n_epochs)
    train_data = ParityDataset(n_samples=args.n_train_samples * int(args.n_elems / 20),
                               n_elems=args.n_elems,
                               noisy_label=noisy_label,
                               exclude_dataset=None,
                               unique=True,
                               model='rnn',
                               noise=args.noise)
    test_data = ParityDataset(n_samples=args.n_eval_samples,
                              n_elems=args.n_elems,
                              noisy_label=noisy_label,
                              exclude_dataset=train_data,
                              unique=True,
                              model='rnn',
                              noise=args.noise)

    train_dataloader = DataLoader(train_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)

    for X_batch, y_batch in train_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred_batch = lstm_model(X_batch)[:, 0]

        loss = criterion(y_pred_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = batch_accuracy(y_pred_batch, y_batch)
        if train_acc > train_max_acc:
            train_max_acc = train_acc

        if (num_steps % eval_interval) == 0:
            val_acc = dataloader_accuracy(test_dataloader, lstm_model)
            print(val_acc)
            if val_acc > 0.95:
                with open(f"{result_folder}/n={args.n_elems}.txt", "a") as f:
                    f.write(f"Test: {val_acc}, Train: {train_max_acc}, Steps: {num_steps}\n")
                sys.exit()

        num_steps += 1

