import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed
from utils import Dataset
from utils import batch_accuracy, dataloader_accuracy
from models import LSTM

set_seed(30)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_piles = 7
batch_size = 128
eval_interval = 50
num_layers = 3
n_epochs = 500

train_dataset = Dataset(n_samples=10000, n_piles=n_piles)
test_dataset = Dataset(n_samples=2000, n_piles=n_piles)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
dataloader_dict = {
    'test': DataLoader(test_dataset, batch_size=batch_size),
}

lstm_model = LSTM(input_size=1,
                  hidden_size=128,
                  num_layers=num_layers)
lstm_model = lstm_model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
writer = SummaryWriter(f'logs/{n_piles}_{num_layers}')


num_steps = 0
for num_epoch in range(n_epochs):
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


