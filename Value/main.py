import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed
from utils import ParityDataset, PureParityDataset
from utils import batch_accuracy, dataloader_accuracy
from models import LSTM
import argparse
import os, sys


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--n_elems',
                    type=int,
                    default=20,
                    help='length of the bitstring.')
PARSER.add_argument('--n_train_elems',
                    type=int,
                    default=20,
                    help='length of the bitstring used for training.')
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
                    default=10000,
                    help='Number of epochs to train.')
PARSER.add_argument('--n_layers',
                    type=int,
                    default=1,
                    help='Number of layers.')
PARSER.add_argument('--train_unique',
                    type=bool,
                    default='',
                    help='if the training dataset contains duplicated data.')
PARSER.add_argument('--noise',
                    type=bool,
                    default='.',
                    help='if the parity data contain noise')
PARSER.add_argument('--log_folder',
                    type=str,
                    default='logs',
                    help='log folder')
PARSER.add_argument('--seed',
                    type=int,
                    default=0,
                    help='seed')
PARSER.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='learning rate')


args = PARSER.parse_args()
print(args)

set_seed(args.seed)

result_folder = "approach_1"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

train_data = PureParityDataset(n_samples=args.n_train_samples * int(args.n_elems/20),
                           n_elems=args.n_elems,
                           n_nonzero_min=1,
                           n_nonzero_max=args.n_train_elems,
                           exclude_dataset=None,
                           unique=args.train_unique,
                           model='rnn',
                           noise=args.noise)
val_data = PureParityDataset(n_samples=args.n_eval_samples,
                         n_elems=args.n_elems,
                         n_nonzero_min=1,
                         n_nonzero_max=args.n_train_elems,
                         exclude_dataset=train_data,
                         unique=True,
                         model='rnn',
                         noise=args.noise)
# extra_data = PureParityDataset(n_samples=args.n_eval_samples,
#                            n_elems=args.n_elems+10,
#                            n_nonzero_min=args.n_elems,
#                            n_nonzero_max=args.n_elems+10,
#                            exclude_dataset=None,
#                            unique=True,
#                            model='rnn',
#                            noise=args.noise)

batch_size = 128
train_dataloader = DataLoader(train_data, batch_size=batch_size)
dataloader_dict = {
    'validation': DataLoader(val_data, batch_size=batch_size),
    # 'extra': DataLoader(extra_data, batch_size=batch_size),
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_interval = 100
lstm_model = LSTM(input_size=1,
                  hidden_size=56,
                  num_layers=args.n_layers)
lstm_model = lstm_model.to(device)

criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.0001 * args.lr
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
# writer = SummaryWriter(f'{args.log_folder}/{args.n_elems}/{args.lr}_{args.n_train_samples}')


num_steps = 0
for num_epoch in range(args.n_epochs):
    print(f'Epochs: {num_epoch}')
    for X_batch, y_batch in train_dataloader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred_batch = lstm_model(X_batch)[:, 0]
        # train_batch_acc = batch_accuracy(y_pred_batch, y_batch)
        # writer.add_scalar('train_batch_accuracy', train_batch_acc, num_steps)

        loss = criterion(y_pred_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (num_steps % eval_interval) == 0:
            for loader_name, loader in dataloader_dict.items():
                val_acc = dataloader_accuracy(loader, lstm_model)
                # print(val_acc)
                # writer.add_scalar(loader_name, val_acc, num_steps)
                if val_acc > 0.95:
                    with open(f"{result_folder}/n={args.n_elems}.txt", "a") as f:
                        f.write(f"{val_acc}-{num_steps}\n")
                    sys.exit()

        num_steps += 1

# torch.save(lstm_model.state_dict(), f'models/{args.n_elems}_{args.lr}_{args.n_train_samples}.pt')

