"""
Warm-start utilities for CNN fine-tuning from a previous checkpoint.

Two-phase strategy:
  Phase 1 (head_freeze_epochs epochs): only head.* params trained at head_lr
  Phase 2 (remaining epochs):          all params trained at initial_lr
"""

import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, NAdam, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR

_OPT_MAP = {'adam': Adam, 'adamw': AdamW, 'rmsprop': RMSprop, 'nadam': NAdam}


def _load_checkpoint_weights(
    model: nn.Module, ckpt_path: str, device: torch.device
) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return ckpt


def _freeze_to_head_only(model: nn.Module) -> int:
    """Freeze all params except head.*. Returns trainable param count."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('head')
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def setup_warmstart(
    model: nn.Module,
    warmstart_ckpt_paths: Optional[dict],
    fold: int,
    device: torch.device,
    head_freeze_epochs: int,
    head_lr: float,
    optimizer_name: str,
    weight_decay: float,
) -> Tuple[bool, Optional[torch.optim.Optimizer]]:
    """Load checkpoint and optionally set up phase-1 (head-only) training.

    Returns:
        phase1_active: True if head-freeze phase should be used this fold.
        phase1_optimizer: head-only Optimizer if phase1_active, else None.
    """
    if not (warmstart_ckpt_paths and fold in warmstart_ckpt_paths):
        return False, None

    ckpt_path = warmstart_ckpt_paths[fold]
    ckpt = _load_checkpoint_weights(model, ckpt_path, device)
    print(f"  Warm-start: {os.path.basename(ckpt_path)} "
          f"(epoch={ckpt.get('epoch', '?')})")

    if head_freeze_epochs <= 0:
        return False, None

    n_head = _freeze_to_head_only(model)
    print(f"  Phase 1: head-only for {head_freeze_epochs} epochs "
          f"(lr={head_lr:.1e}, {n_head:,} params)")

    opt_cls = _OPT_MAP.get(optimizer_name.lower(), Adam)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = opt_cls(head_params, lr=head_lr, weight_decay=weight_decay)
    return True, optimizer


def maybe_transition_phase2(
    epoch: int,
    head_freeze_epochs: int,
    phase1_active: bool,
    model: nn.Module,
    get_optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
    remaining_epochs: int,
    initial_lr: float,
) -> Tuple[bool, Optional[torch.optim.Optimizer], Optional[CosineAnnealingLR]]:
    """At the phase boundary, unfreeze all layers and rebuild optimizer/scheduler.

    Returns updated (phase1_active, optimizer, scheduler).
    If no transition this epoch, returns (phase1_active, None, None).
    """
    if not (phase1_active and epoch == head_freeze_epochs):
        return phase1_active, None, None

    unfreeze_all(model)
    optimizer = get_optimizer_fn(model)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, remaining_epochs), eta_min=initial_lr * 0.001)
    print(f"  Epoch {epoch + 1}: Phase 2 — all layers unfrozen "
          f"(lr={initial_lr:.1e})")
    return False, optimizer, scheduler
