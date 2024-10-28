import random

import numpy as np
import torch
import torch.nn.functional as F


def initialize(seed: int=21, deterministic: bool=True, tf32: bool=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def calculate_metrics(avg_probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 10):

    preds = torch.argmax(avg_probs, dim=1)
    acc = (preds == labels).float().mean()

    nll = F.nll_loss(torch.log(avg_probs), labels)

    ece = torch.zeros(1, device=avg_probs.device)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)

    confidence = avg_probs.max(dim=1).values
    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (
            avg_probs.max(dim=1).values > bin_lower) & (avg_probs.max(dim=1).values <= bin_upper
        )
        if mask.sum() > 0:
            bin_acc = (preds[mask] == labels[mask]).float().mean()
            bin_conf = confidence[mask].mean()
            ece += mask.float().mean() * torch.abs(bin_acc - bin_conf)

    one_hot = F.one_hot(labels, num_classes=avg_probs.size(1))
    # Note that we don't divide the number of classes
    brier_score = (avg_probs - one_hot).pow(2).sum(dim=1).mean()

    return acc.item(), nll.item(), ece.item(), brier_score.item()
