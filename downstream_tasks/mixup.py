import random

import numpy as np
import torch
from cutmix.cutmix import CutMix
from cutmix.utils import onehot, rand_bbox
from torch.utils.data.dataset import Dataset


def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix
            The shape is (C, C), where C is the number of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor), (
        f"pred and target should be torch.Tensor or np.ndarray, "
        f"but got {type(pred)} and {type(target)}."
    )

    # Modified from PyTorch-Ignite
    num_classes = pred.size(1)
    pred_label = torch.argmax(pred, dim=1).flatten()
    target_label = target.flatten()
    assert len(pred_label) == len(target_label)

    with torch.no_grad():
        indices = num_classes * target_label + pred_label
        matrix = torch.bincount(indices, minlength=num_classes**2)
        matrix = matrix.reshape(num_classes, num_classes)
    return matrix.detach().cpu().numpy()


def calculate_accuracy(pred, target):
    _, predicted_labels = torch.max(pred, dim=1)
    correct_predictions = torch.sum(predicted_labels == target)
    total_samples = target.size(0)

    accuracy = correct_predictions.item() / total_samples
    return accuracy


def is_vector_label(x):
    if isinstance(x, np.ndarray):
        return x.size > 1
    elif isinstance(x, torch.Tensor):
        return x.size().numel() > 1
    elif isinstance(x, int):
        return False
    else:
        raise TypeError(f"Unknown type {type(x)}")


class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1.0, prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        example = self.dataset[index]
        img, lb = example["pixel_values"], example["label"]
        lb_onehot = lb if is_vector_label(lb) else onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            rand_example = self.dataset[rand_index]
            img2, lb2 = rand_example["pixel_values"], rand_example["label"]
            lb2_onehot = lb2 if is_vector_label(lb2) else onehot(self.num_class, lb2)
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2])
            )
            lb_onehot = lb_onehot * lam + lb2_onehot * (1.0 - lam)

        return {"pixel_values": img, "label": lb_onehot}

    def __len__(self):
        return len(self.dataset)


def mixup_data(x, y, alpha=1, num_classes=200):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    if is_vector_label(y):
        mixed_y = lam * y + (1 - lam) * y[index]
    else:
        mixed_y = onehot(y, num_classes) * lam + onehot(y[index], num_classes) * (
            1 - lam
        )
    return mixed_x, mixed_y
