import random
import torch
import torch.optim as optim
import torch.nn as nn
from ntm.ntm import NTM
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils import ParityDataset
from utils import batch_accuracy, dataloader_accuracy
from torch.utils.data import DataLoader

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
                    default=128000,
                    help='number of training samples.')
PARSER.add_argument('--n_eval_samples',
                    type=int,
                    default=12800,
                    help='number of evaluation samples')
PARSER.add_argument('--n_epochs',
                    type=int,
                    default=100,
                    help='Number of epochs to train.')
PARSER.add_argument('--memory_size',
                    type=int,
                    default=10,
                    help='memory_size')
PARSER.add_argument('--learning_rate',
                    type=float,
                    default=0.0005,
                    help='learning_rate')
PARSER.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help='batch_size')
PARSER.add_argument('--noise',
                    type=bool,
                    default='.',
                    help='add noise to the parity data')
PARSER.add_argument('--seed',
                    type=int,
                    default=0,
                    help='random seed')
PARSER.add_argument('--log_folder',
                    type=str,
                    default='results',
                    help='log folder')

args = PARSER.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

vector_length = 1
train_data = ParityDataset(n_samples=args.n_train_samples,
                           n_elems=args.n_elems,
                           n_nonzero_min=1,
                           n_nonzero_max=args.n_train_elems,
                           exclude_dataset=None,
                           model='rnn',
                           noise=args.noise)  # (n_train_samples, n_elems, 1)
train_dataloader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True)  # (128, n_elems, 1)
val_data = ParityDataset(n_samples=args.n_eval_samples,
                         n_elems=args.n_elems,
                         n_nonzero_min=1,
                         n_nonzero_max=args.n_train_elems,
                         exclude_dataset=train_data,
                         model='rnn',
                         noise=args.noise)
extra_data = ParityDataset(n_samples=args.n_eval_samples,
                           n_elems=args.n_elems+10,
                           n_nonzero_min=args.n_elems,
                           n_nonzero_max=args.n_elems+10,
                           exclude_dataset=None,
                           model='rnn',
                           noise=args.noise)

dataloader_dict = {
    'validation': DataLoader(val_data, batch_size=args.batch_size),
    'extra': DataLoader(extra_data, batch_size=args.batch_size),
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def train():
    writer = SummaryWriter(f'{args.log_folder}/{args.n_elems}_{args.n_train_elems}_{args.n_train_samples}_'
                           f'{args.learning_rate}_{args.memory_size}_' +
                           f'{args.n_epochs}_{args.batch_size}_{args.noise}_{args.seed}')

    model = NTM(vector_length=vector_length,
                hidden_size=128,
                memory_size=(128, args.memory_size),
                lstm_controller=True)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), momentum=0.9, alpha=0.95, lr=args.learning_rate)
    feedback_frequency = 50
    training_step = 0
    for epoch in range(args.n_epochs):
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            print(training_step)

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            input = torch.zeros(args.n_elems + 1, args.batch_size, vector_length + 1).to(device)
            input[:args.n_elems, :, :vector_length] = X_batch.transpose(0, 1)
            # add delimiter vector
            input[args.n_elems, :, vector_length] = 1.0
            target = y_batch.unsqueeze(dim=1)

            state = model.get_initial_state(batch_size=args.batch_size)
            for vector in input:
                output, state = model(vector, state)
            # output, _ = model(torch.zeros(args.batch_size, vector_length + 1).to(device), state)

            loss = criterion(output, target)
            loss.backward()
            # all gradient components are clipped elementwise to the range (-10, 10).
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10)
            optimizer.step()

            writer.add_scalar('training loss', loss, training_step)
            writer.add_scalar('training accuracy',
                              batch_accuracy(output.view(-1), target.view(-1)),
                              training_step)
            training_step += 1

            if training_step % feedback_frequency == 0:
                for loader_name, loader in dataloader_dict.items():
                    val_acc = dataloader_accuracy(loader, model, args)
                    writer.add_scalar(loader_name, val_acc, training_step)

    torch.save(model.state_dict(), 'models/parity.pt')


if __name__ == "__main__":
    train()
