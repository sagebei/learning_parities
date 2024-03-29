import torch
from torch.utils.data import DataLoader

from parityfunction.utils import ParityDataset
from parityfunction.utils import dataloader_accuracy
import argparse

import numpy as np
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--n_elems',
                    type=int,
                    default=1000,
                    help='length of the bitstring.')
PARSER.add_argument('--n_train_elems',
                    type=int,
                    default=1000,
                    help='length of the bitstring used for training.')
PARSER.add_argument('--n_train_samples',
                    type=int,
                    default=128000,
                    help='number of training samples.')
PARSER.add_argument('--noise',
                    type=bool,
                    default='',
                    help='number of training samples.')


args = PARSER.parse_args()
print(args)

train_data = ParityDataset(n_samples=args.n_train_samples,
                           n_elems=args.n_elems,
                           n_nonzero_min=1,
                           n_nonzero_max=args.n_train_elems,
                           exclude_dataset=None,
                           unique=False,
                           model='rnn',
                           noise=args.noise)


batch_size = 128
train_dataloader = DataLoader(train_data, batch_size=batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


lstm_model = torch.load(f'models/20_{args.noise}.pt').to(device)
acc = dataloader_accuracy(train_dataloader, lstm_model)
print(acc)

