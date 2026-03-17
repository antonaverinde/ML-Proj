"""
Stateless train/validate epoch loops.

All mode-specific logic lives in the Strategy object; these functions contain
zero mode branches.  They accept any BaseForwardStrategy and work uniformly.
"""

from typing import List

import torch
import torch.nn as nn

from .forward_strategies import BaseForwardStrategy


def _compute_metrics(
    prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
):
    """Compute accuracy, precision, recall, IoU from sigmoid prob and binary target."""
    pred = (prob > threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    acc  = (pred == target).float().mean().item()
    prec = (tp / (tp + fp + 1e-6)).item()
    rec  = (tp / (tp + fn + 1e-6)).item()
    iou  = (tp / (tp + fp + fn + 1e-6)).item()
    return acc, prec, rec, iou


def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    strategy: BaseForwardStrategy,
    device: torch.device,
    clip_norm: float = 1.0,
) -> List[float]:
    """Run one training epoch.  Returns [loss, acc, prec, rec, iou]."""
    model.train()
    totals = [0.] * 5
    for batch in loader:
        optimizer.zero_grad()
        logit, xgb_prob, mask = strategy.forward(model, batch, device)
        loss, prob_for_metrics = strategy.training_loss(logit, xgb_prob, mask, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        with torch.no_grad():
            acc, prec, rec, iou = _compute_metrics(prob_for_metrics, mask)
        totals[0] += loss.item()
        totals[1] += acc
        totals[2] += prec
        totals[3] += rec
        totals[4] += iou
    n = len(loader)
    return [v / n for v in totals]


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    strategy: BaseForwardStrategy,
    device: torch.device,
    threshold: float = 0.5,
) -> List[float]:
    """Validation — same combined-logit objective as training for all strategies.

    Returns [loss, acc, prec, rec, iou].
    """
    model.eval()
    totals = [0.] * 5
    for batch in loader:
        logit, xgb_prob, mask = strategy.forward(model, batch, device)
        loss, _ = strategy.training_loss(logit, xgb_prob, mask, criterion)
        prob     = strategy.validation_prob(logit, xgb_prob)
        totals[0] += loss.item()
        acc, prec, rec, iou = _compute_metrics(prob, mask, threshold)
        totals[1] += acc
        totals[2] += prec
        totals[3] += rec
        totals[4] += iou
    n = len(loader)
    return [v / n for v in totals]
