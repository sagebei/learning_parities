import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import ray
import ray.train as train
from ray.train.trainer import Trainer

from utils import ParityDataset
from utils import batch_accuracy, dataloader_accuracy
from models import SelfAttentionModel
import argparse
import numpy as np
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def train_distributed():
    device = torch.device(f'cuda:{train.local_rank()}' if torch.cuda.is_available() else 'cpu')
    print(device)
    writer = SummaryWriter(f'{args.log_folder}/{args.embed_dim}_{args.linear_dim}_{args.n_heads}_{args.mode}' +
                           f'_{args.n_elems}_{args.n_train_elems}_{args.n_layers}_{args.n_epochs}' +
                           f'_{args.n_eval_samples}_{args.n_train_samples}' +
                           f'_{args.train_unique}_{args.n_exclusive_data}_w{args.num_workers}')
    train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler)
    dataloader_dict = {
        'validation': DataLoader(val_data,
                                 batch_size=args.batch_size,
                                 sampler=DistributedSampler(val_data)),
        'extra': DataLoader(extra_data,
                            batch_size=args.batch_size,
                            sampler=DistributedSampler(extra_data)),
    }

    selfattention_model = SelfAttentionModel(n_layers=args.n_layers,
                                             seq_len=args.n_elems,
                                             input_dim=3,
                                             embed_dim=args.embed_dim,
                                             linear_dim=args.linear_dim,
                                             dropout=args.dropout,
                                             n_heads=args.n_heads,
                                             mode=args.mode)
    selfattention_model = selfattention_model.to(device)
    selfattention_model = DistributedDataParallel(selfattention_model,
                                                  device_ids=[device.index] if torch.cuda.is_available() else None)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(selfattention_model.parameters(), lr=0.0003)

    num_steps = 0
    eval_interval = 50
    for num_epoch in range(args.n_epochs):
        train_sampler.set_epoch(num_epoch)
        print(f'Epochs: {num_epoch}')
        for X_batch, y_batch in train_dataloader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred_batch = selfattention_model(X_batch)[:, 0]
            train_batch_acc = batch_accuracy(y_pred_batch, y_batch)
            writer.add_scalar('train_batch_accuracy', train_batch_acc, num_steps * args.num_workers)

            loss = criterion(y_pred_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (num_steps % eval_interval) == 0:
                for loader_name, loader in dataloader_dict.items():
                    val_acc = dataloader_accuracy(loader, selfattention_model)
                    writer.add_scalar(loader_name, val_acc, num_steps * args.num_workers)

            num_steps += 1


def main(num_workers=4, use_gpu=False):
    ray.init()
    trainer = Trainer(backend='torch',
                      num_workers=num_workers,
                      use_gpu=use_gpu)
    trainer.start()
    trainer.run(train_func=train_distributed)
    trainer.shutdown()


if __name__ == '__main__':
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
    PARSER.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='batch size')
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
    PARSER.add_argument('--mode',
                        type=str,
                        default='soft',
                        help='soft or hard')
    PARSER.add_argument('--embed_dim',
                        type=int,
                        default=30,
                        help='embedding dim in the multi-head self attention')
    PARSER.add_argument('--n_heads',
                        type=int,
                        default=3,
                        help='Number of attention heads')
    PARSER.add_argument('--linear_dim',
                        type=int,
                        default=30,
                        help='hidden size of the linear layers')
    PARSER.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='dropout value in linear layers')
    PARSER.add_argument('--log_folder',
                        type=str,
                        default='results',
                        help='log folder')
    PARSER.add_argument('--num_workers',
                        type=int,
                        default=2,
                        help='number of workers')

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
    main(num_workers=args.num_workers,
         use_gpu=False)

