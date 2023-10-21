import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from copy import deepcopy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BoardDataset(Dataset):
    def __init__(self, n_samples, n_piles, existing_X):
        self.n_samples = n_samples
        self.n_piles = n_piles
        self.X = list()
        self.Y = list()
        self.existing_X = deepcopy(existing_X) if len(existing_X) == 0 else np.squeeze(deepcopy(existing_X)).astype(int).tolist()
        self.build_dataset()
        print(Counter(self.Y))

    def __len__(self):
        return len(self.X)

    def generate_data(self):
        x = []
        num_ones = 0
        num_twos = 0
        for n_pile in range(self.n_piles):
            r = random.randint(0, 2)
            if r == 0:
                x.extend([0, 0, -1])
            elif r == 1:
                num_ones += 1
                x.extend([0, 1, -1])
            elif r == 2:
                num_twos += 1
                x.extend([1, 1, -1])

        x = x[:-1]

        num_ones %= 2
        num_twos %= 2

        if x in self.existing_X:
            return None, None

        if num_ones == 0 and num_twos == 0:
            return None, None
        elif num_ones == 1 and num_twos == 0:
            return x, 0
        elif num_ones == 0 and num_twos == 1:
            return x, 1
        elif num_ones == 1 and num_twos == 1:
            return x, 2

    def build_dataset(self):
        n_sample_per_class = (self.n_samples // 3) + 1
        x_zeros = []
        x_ones = []
        x_twos = []

        while True:
            x, y = self.generate_data()
            if x is not None:
                x = np.expand_dims(x, axis=-1).astype(np.float32)

                if y == 0:
                    x_zeros.append(x)
                elif y == 1:
                    x_ones.append(x)
                elif y == 2:
                    x_twos.append(x)

            if len(x_zeros) >= n_sample_per_class and len(x_ones) >= n_sample_per_class and len(x_twos) >= n_sample_per_class:
                break

        self.X.extend(random.sample(x_zeros, n_sample_per_class))
        self.X.extend(random.sample(x_ones, n_sample_per_class))
        self.X.extend(random.sample(x_twos, n_sample_per_class))

        for i in range(3):
            for _ in range(n_sample_per_class):
                self.Y.append(i)

        zipped = list(zip(self.X, self.Y))
        random.shuffle(zipped)
        self.X, self.Y = zip(*zipped)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class RealBoardDataset(BoardDataset):
    def __init__(self, n_samples, n_piles, existing_X):
        super().__init__(n_samples, n_piles, existing_X)

    def generate_data(self):
        x = []
        num_ones = 0
        num_twos = 0
        n_piles = torch.randint(1, self.n_piles, (1,)).item()

        for n_pile in range(n_piles):
            r = random.randint(1, 2)
            if r == 1:
                num_ones += 1
                x.append([0, 1, -1])
            elif r == 2:
                num_twos += 1
                x.append([1, 1, -1])

        for _ in range(self.n_piles - n_piles):
            x.append([0, 0, -1])

        random.shuffle(x)
        X = []
        for i in x:
            X.extend(i)

        X = X[:-1]

        num_ones %= 2
        num_twos %= 2

        if X in self.existing_X:
            return None, None

        if num_ones == 0 and num_twos == 0:
            return None, None
        elif num_ones == 1 and num_twos == 0:
            return X, 0
        elif num_ones == 0 and num_twos == 1:
            return X, 1
        elif num_ones == 1 and num_twos == 1:
            return X, 2

def batch_accuracy(y_pred_batch, y_batch):
    y_pred_batch = torch.argmax(y_pred_batch, dim=1, keepdim=False)
    acc = (y_pred_batch == y_batch).float().mean().item()
    return acc


def dataloader_accuracy(dataloader, model):
    model.eval()
    accuracy = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            batch_acc = batch_accuracy(y_pred, y_batch)
            accuracy.append(batch_acc)
    model.train()
    if len(accuracy) == 0:
        return 0
    return sum(accuracy) / len(accuracy)


if __name__ == '__main__':
    data = RealBoardDataset(n_samples=10, n_piles=4, existing_X=[])
    for x, y in zip(data.X, data.Y):
        print(x, y)
