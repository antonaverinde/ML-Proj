"""
Loss functions for segmentation training (PyTorch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lovasz_loss import lovasz_hinge

class LovaszHingeLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return lovasz_hinge(logits, targets.float(), per_image=True)

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        probs = probs.flatten()
        targets = targets.flatten()

        inter = (probs * targets).sum()
        dice = (2 * inter + self.eps) / (probs.sum() + targets.sum() + self.eps)
        return 1 - dice
class LogDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = DiceLoss(self.eps)(logits, targets)
        return -torch.log(1 - dice_loss + self.eps)
    
class SoftIoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        probs = probs.flatten()
        targets = targets.flatten()

        inter = (probs * targets).sum()
        union = probs.sum() + targets.sum() - inter

        iou = (inter + self.eps) / (union + self.eps)
        return 1 - iou



class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy loss."""

    def __init__(self, pos_w: float, neg_w: float):
        super().__init__()
        self.pos_w = pos_w
        self.neg_w = neg_w

# For logits 

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # pos_weight in PyTorch BCE expects weight for positive class only
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=torch.tensor(self.pos_w, device=logits.device),
            reduction='mean'
        )
        return bce

class CombinedLoss(nn.Module):
    """Combined weighted BCE + Dice loss."""

    def __init__(self, pos_w: float, neg_w: float, alpha: float = 0.4, beta: float = 0.6):
        super().__init__()
        self.wbce = WeightedBCE(pos_w, neg_w)
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.wbce(y_pred, y_true) + self.beta * DiceLoss()(y_pred, y_true)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0,pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
    def forward(self, inputs, targets):
        targets = targets.float()
        pw = self.pos_weight.to(inputs.device) if self.pos_weight is not None else None
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=pw)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        # alpha penalizes FP, beta penalizes FN
        # alpha=0.7, beta=0.3 → punish false positives more
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = targets.float()
        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()
        return 1 - (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
class CombinedLoss2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        
    def focal_loss(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets, smooth=1.0):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid for Dice
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return focal + self.dice_weight * dice


def get_loss_function(loss_name: str, pos_w: float = 1.0, neg_w: float = 1.0,
                      alpha: float = 0.4, beta: float = 0.6):
    """Get loss function by name."""
    loss_name = loss_name.lower()

    if loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'weighted':
        return WeightedBCE(pos_w, neg_w)
    elif loss_name == 'focal':
        return FocalLoss(alpha=alpha, gamma=beta,pos_weight=pos_w)#focal_loss
    elif loss_name == 'soft_iou':
        return SoftIoULoss()
    elif loss_name == 'combined':
        return CombinedLoss(pos_w, neg_w, alpha, beta)
    elif loss_name == 'combined2':
        return CombinedLoss2(alpha=alpha, gamma=beta,dice_weight=1.0)
    elif loss_name == "log_dice":
        return LogDiceLoss()
    elif loss_name == "tversky":
        return TverskyLoss(alpha=alpha, beta=beta)
    elif loss_name == 'lovasz':
      return LovaszHingeLoss()
    else:
        return nn.BCELoss()
#Sigmoid versions (old):
# def soft_iou_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#     """Soft IoU loss."""
#     y_true = y_true.float().flatten()
#     y_pred = y_pred.flatten()
#     inter = (y_true * y_pred).sum()
#     union = y_true.sum() + y_pred.sum() - inter
#     return 1 - (inter + 1e-6) / (union + 1e-6)


# def focal_loss(y_pred: torch.Tensor, y_true: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
#     """Focal loss for class imbalance."""
#     y_true = y_true.float()
#     y_pred = y_pred.clamp(1e-7, 1 - 1e-7)
#     p_t = torch.where(y_true == 1, y_pred, 1 - y_pred)
#     fl = -alpha * (1 - p_t).pow(gamma) * p_t.log()
#     return fl.mean()



# def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#     """Dice loss for binary segmentation."""
#     y_true = y_true.float().flatten()
#     y_pred = y_pred.flatten()
#     inter = (y_true * y_pred).sum()
#     return 1 - (2 * inter + 1e-6) / (y_true.sum() + y_pred.sum() + 1e-6)
# def log_dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#     """Log Dice loss for binary segmentation."""
#     y_true = y_true.float().flatten()
#     y_pred = y_pred.flatten()
#     dice = dice_loss(y_pred, y_true)
#     return -torch.log(1 - dice + 1e-6)

# For sigmoid (probabilities)
    # def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    #     y_true = y_true.float()
    #     y_pred = y_pred.clamp(1e-7, 1 - 1e-7)
    #     bce = -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log())
    #     weights = y_true * self.pos_w + (1 - y_true) * self.neg_w
    #     return (weights * bce).mean()