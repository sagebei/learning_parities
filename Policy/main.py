import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed, BoardDataset
from utils import batch_accuracy, dataloader_accuracy
from models import LSTM
import argparse

set_seed(30)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--n_piles',
                    type=int,
                    default=8,
                    help='length of the bitstring.')
PARSER.add_argument('--num_layers',
                    type=int,
                    default=1,
                    help='length of the bitstring used for training.')
PARSER.add_argument('--n_train_samples',
                    type=int,
                    default=5000,
                    help='number of training samples.')
PARSER.add_argument('--n_test_samples',
                    type=int,
                    default=1000,
                    help='number of training samples.')
PARSER.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='number of training samples.')
PARSER.add_argument('--n_epochs',
                    type=int,
                    default=2000,
                    help='number of training samples.')

args = PARSER.parse_args()
batch_size = 64
eval_interval = 50

train_dataset = BoardDataset(n_samples=args.n_train_samples, n_piles=args.n_piles, existing_X=[])
test_dataset = BoardDataset(n_samples=args.n_test_samples, n_piles=args.n_piles, existing_X=train_dataset.X)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_dict = {
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=True),
}

lstm_model = LSTM(input_size=1,
                  hidden_size=128,
                  num_layers=args.num_layers)
lstm_model = lstm_model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=args.lr)
writer = SummaryWriter(f'/data/scratch/acw554/parity/policy/{args.n_piles}_{args.num_layers}_{args.lr}_{args.n_train_samples}_{args.n_test_samples}')


num_steps = 0
for num_epoch in range(args.n_epochs):
    print(f'Epochs: {num_epoch}')
    for X_batch, y_batch in train_dataloader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred_batch = lstm_model(X_batch)
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


