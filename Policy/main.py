import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed, BoardDataset
from utils import batch_accuracy, dataloader_accuracy
from models import LSTM
import argparse
import sys
import os

set_seed(369)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--n_piles',
                    type=int,
                    default=7,
                    help='length of the bitstring.')
PARSER.add_argument('--num_layers',
                    type=int,
                    default=1,
                    help='length of the bitstring used for training.')
PARSER.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='number of training samples.')
args = PARSER.parse_args()

total_samples = 3 ** args.n_piles
n_train_samples = min(int(total_samples * 0.7), 1000000)
n_test_samples = min(int(total_samples * 0.1), 10000)

batch_size = 128
eval_interval = 100

train_dataset = BoardDataset(n_samples=n_train_samples, n_piles=args.n_piles, existing_X=[])
test_dataset = BoardDataset(n_samples=n_test_samples, n_piles=args.n_piles, existing_X=train_dataset.X)

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
# writer = SummaryWriter(f'/data/scratch/acw554/parity/policy/{args.n_piles}_{args.num_layers}_{args.lr}_{args.n_train_samples}_{args.n_test_samples}')


if not os.path.exists("converge"):
    os.makedirs("converge")

num_steps = 0
while num_steps < 1000000:
    for X_batch, y_batch in train_dataloader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred_batch = lstm_model(X_batch)
        # train_batch_acc = batch_accuracy(y_pred_batch, y_batch)
        # print(train_batch_acc)
        # writer.add_scalar('train_batch_accuracy', train_batch_acc, num_steps)

        loss = criterion(y_pred_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (num_steps % eval_interval) == 0:
            for loader_name, loader in dataloader_dict.items():
                val_acc = dataloader_accuracy(loader, lstm_model)
                # writer.add_scalar(loader_name, val_acc, num_steps)
                if val_acc > 0.90:
                    with open(f"converge/{args.n_piles}.txt", "a") as f:
                        f.write(f"{val_acc}-{num_steps}\n")
                    sys.exit()

        num_steps += 1


