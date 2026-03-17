"""
Threshold search for binary segmentation.

Finds the threshold in [0.10, 0.65) that maximises IoU on the validation set.
Uses validation_prob from the strategy so the threshold matches inference behaviour.
"""

import torch

from .forward_strategies import BaseForwardStrategy


def find_best_threshold(
    model: torch.nn.Module,
    loader,
    strategy: BaseForwardStrategy,
    device: torch.device,
) -> float:
    """Vectorised IoU-maximising search over thresholds [0.10, 0.65).

    Returns the threshold (float) that achieves the highest mean IoU.
    """
    model.eval()
    all_probs, all_masks = [], []
    with torch.no_grad():
        for batch in loader:
            logit, xgb_prob, mask = strategy.forward(model, batch, device)
            prob = strategy.validation_prob(logit, xgb_prob)
            all_probs.append(prob.cpu())
            all_masks.append(mask.cpu())

    probs  = torch.cat(all_probs)    # (N, H, W)
    masks  = torch.cat(all_masks)
    thrs   = torch.arange(0.10, 0.65, 0.05)
    preds  = (probs.unsqueeze(0) > thrs[:, None, None, None]).float()
    target = masks.unsqueeze(0)
    tp = (preds * target).sum(dim=(1, 2, 3))
    fp = (preds * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * target).sum(dim=(1, 2, 3))
    return thrs[(tp / (tp + fp + fn + 1e-6)).argmax()].item()
