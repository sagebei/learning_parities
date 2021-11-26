import torch
import torch.nn as nn
from torch.utils.data import Dataset
from os.path import exists


class ParityDataset(Dataset):
    def __init__(
        self,
        n_samples,
        n_elems,
        n_nonzero_min=None,
        n_nonzero_max=None,
        exclude_dataset=None,
        unique=False,
        model='rnn',
        data_augmentation=0,
    ):
        self.n_samples = n_samples
        self.n_elems = n_elems

        self.n_nonzero_min = 1 if n_nonzero_min is None else n_nonzero_min
        self.n_nonzero_max = (
            n_elems if n_nonzero_max is None else n_nonzero_max
        )

        self.model = model
        self.data_augmentation = data_augmentation
        self.unique = unique
        self.unique_set = set()
        self.val_set = set() if exclude_dataset is None else exclude_dataset.unique_set

        self.X, self.Y = [], []
        dataset_path = f"../datasets/{n_samples}_{n_elems}_{n_nonzero_max}_{n_nonzero_min}_{unique}_{model}.pt"
        if exists(dataset_path):
            self.X, self.Y = torch.load(dataset_path)
        elif self.n_samples > 0:
            if not exists('../datasets'):
                os.mkdir('../datasets')
            self.build_dataset()
            torch.save([self.X, self.Y], dataset_path)

    def __len__(self):
        return self.n_samples

    def generate_data(self):
        while True:
            x = torch.zeros((self.n_elems,))
            n_non_zero = torch.randint(
                self.n_nonzero_min, self.n_nonzero_max + 1, (1,)
            ).item()
            x[:n_non_zero] = torch.randint(0, 2, (n_non_zero,)) * 2 - 1
            x = x[torch.randperm(self.n_elems)]

            y = (x == 1.0).sum() % 2

            item = tuple(x.detach().clone().tolist())
            if self.unique:
                if (item not in self.val_set) and (item not in self.unique_set):
                    return x, y.item(), item
            else:
                if item not in self.val_set:
                    return x, y.item(), item

    def build_dataset(self):
        print('Building dataset ...')
        for _ in range(self.n_samples):
            x, y, item = self.generate_data()
            self.X.append(x)
            self.Y.append(y)
            self.unique_set.add(item)

        if self.data_augmentation > 0:
            n_aug = int(self.data_augmentation * self.n_samples)
            self.X += self.X[:n_aug]
            self.Y += self.Y[:n_aug]
            self.n_samples += n_aug

        self.Y = torch.Tensor(self.Y).float()
        if self.model == 'rnn':
            self.X = torch.stack(self.X).float().unsqueeze(dim=-1)
        elif self.model == 'mlp':
            self.X = torch.stack(self.X).float()
        elif self.model == 'cnn':
            self.X = torch.stack(self.X).to(torch.int64) + 1
            self.X = F.one_hot(self.X, num_classes=3).float()

        perm_index = torch.randperm(self.X.size()[0])
        self.X = self.X[perm_index]
        self.Y = self.Y[perm_index]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class PonderNet(nn.Module):
    def __init__(self, n_elems, n_hidden=64, max_steps=20, allow_halting=False):
        super().__init__()

        self.max_steps = max_steps
        self.n_hidden = n_hidden
        self.allow_halting = allow_halting

        self.cell = nn.GRUCell(n_elems, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 1)
        self.lambda_layer = nn.Linear(n_hidden, 1)

    def forward(self, x):
        batch_size, _ = x.shape
        device = x.device

        h = x.new_zeros(batch_size, self.n_hidden)

        un_halted_prob = x.new_ones(batch_size)

        y_list = []
        p_list = []

        halting_step = torch.zeros(batch_size, dtype=torch.long, device=device)

        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = x.new_ones(batch_size)  # (batch_size,)
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(h))[:, 0]  # (batch_size,)

            y_list.append(self.output_layer(h)[:, 0])  # (batch_size,)
            p_list.append(un_halted_prob * lambda_n)  # (batch_size,)

            halting_step = torch.maximum(
                n * (halting_step == 0) * torch.bernoulli(lambda_n).to(torch.long),
                halting_step,
            )

            # Prepare for next iteration
            un_halted_prob = un_halted_prob * (1 - lambda_n)
            h = self.cell(x, h)

            # Potentially stop if all samples halted
            if self.allow_halting and (halting_step > 0).sum() == batch_size:
                break

        y = torch.stack(y_list)
        p = torch.stack(p_list)

        return y, p, halting_step


class ReconstructionLoss(nn.Module):

    def __init__(self, loss_func):
        super().__init__()

        self.loss_func = loss_func

    def forward(self, p, y_pred, y_true):
        max_steps, _ = p.shape
        total_loss = p.new_tensor(0.0)

        for n in range(max_steps):
            loss_per_sample = p[n] * self.loss_func(
                y_pred[n], y_true
            )  # (batch_size,)
            total_loss = total_loss + loss_per_sample.mean()  # (1,)

        return total_loss


class RegularizationLoss(nn.Module):

    def __init__(self, lambda_p, max_steps=20):
        super().__init__()

        p_g = torch.zeros((max_steps,))
        not_halted = 1.0

        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        self.register_buffer("p_g", p_g)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, p):

        steps, batch_size = p.shape

        p = p.transpose(0, 1)  # (batch_size, max_steps)

        p_g_batch = self.p_g[None, :steps].expand_as(
            p
        )  # (batch_size, max_steps)

        return self.kl_div(p.log(), p_g_batch)
