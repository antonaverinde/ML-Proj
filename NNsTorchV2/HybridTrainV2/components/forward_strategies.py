"""
Forward strategies — one class per hybrid mode.

Each strategy encapsulates:
  1. How to build the model input from a batch (data, xgb_prob, mask).
  2. How to compute training loss (including fusion-weight blending for parallel).
  3. How to compute the probability tensor used for validation metrics.

Adding a new mode means adding a new Strategy subclass here — no other file changes.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

from .hybrid_models import FusionWeight


class BaseForwardStrategy(ABC):
    """Knows how to route a batch through a model for one specific hybrid mode."""

    @abstractmethod
    def forward(
        self, model: nn.Module, batch: Tuple, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack batch, move to device, run model.

        Returns (logit [N,H,W], xgb_prob [N,1,H,W], mask [N,H,W]).
        """

    @abstractmethod
    def training_loss(
        self,
        logit: torch.Tensor,
        xgb_prob: torch.Tensor,
        mask: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and probability tensor for metrics.

        Returns (loss, prob_for_metrics) where prob_for_metrics is sigmoid output.
        """

    @abstractmethod
    def validation_prob(
        self, logit: torch.Tensor, xgb_prob: torch.Tensor
    ) -> torch.Tensor:
        """Return probability tensor (sigmoid already applied) for metrics."""


# ── Concrete strategies ───────────────────────────────────────────────────────

class ProbOnlyStrategy(BaseForwardStrategy):
    """CNN input: XGB probability map only (1 channel)."""

    def forward(self, model, batch, device):
        data, xgb_prob, mask = [b.to(device) for b in batch]
        logit = model(xgb_prob).squeeze(1)
        return logit, xgb_prob, mask

    def training_loss(self, logit, xgb_prob, mask, criterion):
        loss = criterion(logit, mask)
        return loss, torch.sigmoid(logit).detach()

    def validation_prob(self, logit, xgb_prob):
        return torch.sigmoid(logit)


class ProbFeatStrategy(BaseForwardStrategy):
    """CNN input: raw features concatenated with XGB probability (C+1 channels)."""

    def forward(self, model, batch, device):
        data, xgb_prob, mask = [b.to(device) for b in batch]
        feat = torch.cat([data, xgb_prob], dim=1)
        logit = model(feat).squeeze(1)
        return logit, xgb_prob, mask

    def training_loss(self, logit, xgb_prob, mask, criterion):
        loss = criterion(logit, mask)
        return loss, torch.sigmoid(logit).detach()

    def validation_prob(self, logit, xgb_prob):
        return torch.sigmoid(logit)


class ParallelStrategy(BaseForwardStrategy):
    """CNN trained on raw features; combined with XGB prob at inference.

    During training and mid-training validation, loss is computed on the
    combined logit (w * cnn_logit + (1-w) * xgb_logit) so that FusionWeight
    receives gradients.  Final validation uses combined probabilities so that
    metrics reflect the true inference behaviour.
    """

    def __init__(self, fusion: FusionWeight, combine: str = 'mean'):
        self.fusion  = fusion   # may be None when FusionWeight is not used
        self.combine = combine

    def forward(self, model, batch, device):
        data, xgb_prob, mask = [b.to(device) for b in batch]
        logit = model(data).squeeze(1)
        return logit, xgb_prob, mask

    def training_loss(self, logit, xgb_prob, mask, criterion):
        if self.fusion is not None:
            xgb_logit = torch.logit(xgb_prob.squeeze(1).clamp(1e-6, 1 - 1e-6))
            w = self.fusion()
            combined_logit = w * logit + (1.0 - w) * xgb_logit
            loss = criterion(combined_logit, mask)
            return loss, torch.sigmoid(combined_logit).detach()
        loss = criterion(logit, mask)
        return loss, torch.sigmoid(logit).detach()

    def validation_prob(self, logit, xgb_prob):
        """Mid-training validation: same combined-logit signal as training."""
        if self.fusion is not None:
            xgb_logit = torch.logit(xgb_prob.squeeze(1).clamp(1e-6, 1 - 1e-6))
            w = self.fusion()
            return torch.sigmoid(w * logit + (1.0 - w) * xgb_logit)
        return torch.sigmoid(logit)



class NNOnlyStrategy(BaseForwardStrategy):
    """Pure CNN — XGB probability is ignored (zeros in dataset)."""

    def forward(self, model, batch, device):
        data, xgb_prob, mask = [b.to(device) for b in batch]
        logit = model(data).squeeze(1)
        return logit, xgb_prob, mask

    def training_loss(self, logit, xgb_prob, mask, criterion):
        loss = criterion(logit, mask)
        return loss, torch.sigmoid(logit).detach()

    def validation_prob(self, logit, xgb_prob):
        return torch.sigmoid(logit)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_strategy(
    mode: str,
    fusion: FusionWeight = None,
    combine: str = 'mean',
) -> BaseForwardStrategy:
    """Return the correct strategy for *mode*.  fusion is only used for 'parallel'."""
    if mode == 'prob_only':
        return ProbOnlyStrategy()
    if mode == 'prob_feat':
        return ProbFeatStrategy()
    if mode == 'parallel':
        return ParallelStrategy(fusion=fusion, combine=combine)
    if mode == 'nn_only':
        return NNOnlyStrategy()
    raise ValueError(f"Unknown mode {mode!r}")
