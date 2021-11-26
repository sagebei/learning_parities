import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import ParityDataset, PonderNet, ReconstructionLoss, RegularizationLoss


@torch.no_grad()
def evaluate(dataloader, model):
    model.allow_halting = True
    param = next(model.parameters())
    device, dtype = param.device, param.dtype

    metrics_single_ = {"accuracy_halted": [], "halting_step": []}

    for x_batch, y_true_batch in dataloader:
        x_batch = x_batch.to(device, dtype)  # (batch_size, n_elems)
        y_true_batch = y_true_batch.to(device, dtype)  # (batch_size,)

        y_pred_batch, p, halting_step = model(x_batch)
        y_halted_batch = y_pred_batch.gather(
            dim=0,
            index=halting_step[None, :] - 1,
        )[0]  # (batch_size,)

        # Computing single metrics (mean over samples in the batch)
        accuracy_halted = (
            ((y_halted_batch > 0) == y_true_batch).to(torch.float32).mean()
        )
        metrics_single_["accuracy_halted"].append(accuracy_halted)
        metrics_single_["halting_step"].append(halting_step.to(torch.float).mean())

    metrics = {
        name: torch.stack(values).mean(dim=0).cpu().numpy()
        for name, values in metrics_single_.items()
    }

    return metrics


def get_train_accuracy(y_pred_batch, halting_step, y_true_batch):
    y_halted_batch = y_pred_batch.gather(
            dim=0,
            index=halting_step.unsqueeze(0) - 1,
        )[0]  # (batch_size,)

    accuracy = ((y_halted_batch > 0) == y_true_batch).to(torch.float32).mean()
    return accuracy


def main():
    import numpy as np
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

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
    PARSER.add_argument('--train_unique',
                        type=bool,
                        default='',
                        help='if the training dataset contains duplicated data.')
    PARSER.add_argument('--n_exclusive_data',
                        type=int,
                        default=0,
                        help='number of data that the training data does not contain.')
    PARSER.add_argument('--data_augmentation',
                        type=float,
                        default=0,
                        help='Augment the dataset by the specified ratio')
    PARSER.add_argument('--log_folder',
                        type=str,
                        default='results',
                        help='log folder')

    args = PARSER.parse_args()
    print(args)

    writer = SummaryWriter(f'{args.log_folder}/{args.n_elems}_{args.n_train_elems}' +
                           f'_{args.n_epochs}_{args.n_eval_samples}_{args.n_train_samples}' +
                           f'_{args.train_unique}-{args.n_exclusive_data}-{args.data_augmentation}')

    exclusive_data = ParityDataset(n_samples=args.n_exclusive_data,
                                   n_elems=args.n_elems,
                                   n_nonzero_min=1,
                                   n_nonzero_max=args.n_train_elems,
                                   exclude_dataset=None,
                                   unique=True,
                                   model='mlp')
    train_data = ParityDataset(n_samples=args.n_train_samples,
                               n_elems=args.n_elems,
                               n_nonzero_min=1,
                               n_nonzero_max=args.n_train_elems,
                               exclude_dataset=exclusive_data,
                               unique=args.train_unique,
                               model='mlp',
                               data_augmentation=args.data_augmentation)
    val_data = ParityDataset(n_samples=args.n_eval_samples,
                             n_elems=args.n_elems,
                             n_nonzero_min=1,
                             n_nonzero_max=args.n_train_elems,
                             exclude_dataset=train_data,
                             unique=True,
                             model='mlp')
    extra_data = ParityDataset(n_samples=args.n_eval_samples if args.n_elems != args.n_train_elems else 0,
                               n_elems=args.n_elems,
                               n_nonzero_min=args.n_train_elems,
                               n_nonzero_max=args.n_elems,
                               exclude_dataset=None,
                               unique=True,
                               model='mlp')

    batch_size = 128
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    dataloader_dict = {
        'validation': DataLoader(val_data, batch_size=batch_size),
        'extrapolation': DataLoader(extra_data, batch_size=batch_size),
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Model preparation
    model = PonderNet(n_elems=args.n_elems, n_hidden=128, max_steps=20, allow_halting=False)
    model = model.to(device, dtype=torch.float32)

    # Loss preparation
    loss_rec_inst = ReconstructionLoss(nn.BCEWithLogitsLoss(reduction="none")).to(device, dtype=torch.float32)
    loss_reg_inst = RegularizationLoss(lambda_p=0.2, max_steps=20).to(device, dtype=torch.float32)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Training and evaluation loops
    step = 0
    for _ in range(args.n_epochs):
        for x_batch, y_true_batch in train_dataloader:
            model.train()
            model.allow_halting = False
            
            x_batch = x_batch.to(device, dtype=torch.float32)
            y_true_batch = y_true_batch.to(device, dtype=torch.float32)

            y_pred_batch, p, halting_step = model(x_batch)

            loss_rec = loss_rec_inst(p, y_pred_batch, y_true_batch)
            loss_reg = loss_reg_inst(p)

            loss_overall = loss_rec + 0.01 * loss_reg
            train_batch_accuracy = get_train_accuracy(y_pred_batch, halting_step, y_true_batch)

            optimizer.zero_grad()
            loss_overall.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # Logging
            writer.add_scalar('train_batch_accuracy', train_batch_accuracy, step)
            writer.add_scalar("loss_rec", loss_rec, step)
            writer.add_scalar("loss_reg", loss_reg, step)
            writer.add_scalar("loss_overall", loss_overall, step)
            step += 1

            # Evaluation
            if step % 500 == 0:
                model.eval()
                for dataloader_name, dataloader in dataloader_dict.items():
                    metrics_single = evaluate(dataloader, model)
                    for metric_name, metric_value in metrics_single.items():
                        writer.add_scalar(f"{metric_name}/{dataloader_name}", metric_value, step)


if __name__ == "__main__":
    main()
