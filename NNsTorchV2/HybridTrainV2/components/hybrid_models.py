"""
CNN architectures for Hybrid XGBoost+CNN segmentation (HybridTrainV2).

Three modes — all use a pretrained XGBoost:

  'prob_only'  — CNN input: XGB probability map only (1 channel)
                 Learns to refine the spatial structure of the XGB output.

  'prob_feat'  — CNN input: raw features + XGB probability (C+1 channels)
                 Combines spectral information with XGB knowledge.

  'parallel'   — CNN input: raw features only (C channels), trained independently.
                 At inference, CNN probability and XGB probability are combined
                 (mean / max) for the final prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch, out_ch, kernel=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True))


class RefinementCNN(nn.Module):
    """
    Four-layer CNN for segmentation.
    Works for all three hybrid modes — only `in_channels` differs.

    Input  : (in_channels, H, W)
    Output : raw logit (1, H, W)
    """

    def __init__(self, in_channels: int, n_filters: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn_relu(in_channels, n_filters),
            _conv_bn_relu(n_filters, n_filters),
            _conv_bn_relu(n_filters, n_filters // 2),
            nn.Conv2d(n_filters // 2, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Class-imbalance prior: sigmoid(-2.2) ≈ 0.10
        nn.init.constant_(self.net[-1].bias, -2.2)

    def forward(self, x):
        return self.net(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


# ── Version 1: SE only ────────────────────────────────────────────────────────

class RefinementCNNSE(nn.Module):
    def __init__(self, in_channels: int, n_filters: int = 32):
        super().__init__()
        self.block1 = nn.Sequential(_conv_bn_relu(in_channels, n_filters),
                                    SEBlock(n_filters))
        self.block2 = nn.Sequential(_conv_bn_relu(n_filters, n_filters),
                                    SEBlock(n_filters))
        self.block3 = nn.Sequential(_conv_bn_relu(n_filters, n_filters // 2),
                                    SEBlock(n_filters // 2))
        self.head   = nn.Conv2d(n_filters // 2, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.head.bias, -2.2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


# ── Version 2: SE + skip connections ─────────────────────────────────────────

class SEResBlock(nn.Module):
    """Conv-BN-ReLU → Conv-BN → SE → residual add."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = _conv_bn_relu(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.se    = SEBlock(channels, reduction)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.se(self.conv2(self.conv1(x))) + x)


class RefinementCNNSkip(nn.Module):
    def __init__(self, in_channels: int, n_filters: int = 32):
        super().__init__()
        # Stem: project input to n_filters
        self.stem   = _conv_bn_relu(in_channels, n_filters)

        # Two SE-residual blocks at full width
        self.res1   = SEResBlock(n_filters)
        self.res2   = SEResBlock(n_filters)

        # Bottleneck down to n_filters // 2 (no skip — channel mismatch)
        self.down   = nn.Sequential(
            _conv_bn_relu(n_filters, n_filters // 2),
            SEBlock(n_filters // 2)
        )
        self.head   = nn.Conv2d(n_filters // 2, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.head.bias, -2.2)

    def forward(self, x):
        x = self.stem(x)   # (B, n_filters, H, W)
        x = self.res1(x)   # skip inside SEResBlock
        x = self.res2(x)   # skip inside SEResBlock
        x = self.down(x)   # (B, n_filters//2, H, W)
        return self.head(x)


class RefinementMLP(nn.Module):
    """
    Pixel-wise MLP refinement — treats each pixel independently.
    Input  : (B, in_channels, H, W)
    Output : raw logit (B, 1, H, W)
    """

    def __init__(self, in_channels: int, hidden: tuple = (64, 32, 32)):
        super().__init__()

        layers = []
        prev = in_channels
        for h in hidden:
            layers += [nn.Linear(prev, h, bias=False),
                       nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.mlp = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # class-imbalance prior on the last linear bias
        nn.init.constant_(self.mlp[-1].bias, -2.2)

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, C, H, W) → (B*H*W, C)  — treat every pixel as one sample
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = self.mlp(x)                   # (B*H*W, 1)
        x = x.reshape(B, 1, H, W)
        return x

# ── Shared SE for 1D ─────────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int):
        super().__init__()
        self.conv1   = nn.Conv1d(hidden, hidden, kernel_size=3,
                                 dilation=dilation, padding=dilation)
        self.conv2   = nn.Conv1d(hidden, hidden, kernel_size=3,
                                 dilation=dilation, padding=dilation)
        self.norm1   = nn.BatchNorm1d(hidden)
        self.norm2   = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(0.2)
        self.se      = SEBlock1d(hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        x = self.se(x)
        return F.relu(x + residual)
class SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.se(x).unsqueeze(-1)
        return x * w


class TCNEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 32, out_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Conv1d(in_ch, hidden, kernel_size=1)
        self.blocks     = nn.ModuleList([
            TCNBlock(hidden, dilation=d) for d in [1, 2, 4, 8, 16]
        ])
        self.output_proj = nn.Sequential(
            SEBlock1d(hidden),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, out_dim)
        )
        self.head = nn.Linear(out_dim, 1)   # pixel logit
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.head.bias, -2.2)  # class-imbalance prior

    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     x = x.permute(0, 2, 3, 1).reshape(-1, C).unsqueeze(1)  # (B*H*W, 1, C)
    #     x = self.input_proj(x)
    #     for block in self.blocks:
    #         x = block(x)
    #     x = self.output_proj(x)          # (B*H*W, out_dim)
    #     x = self.head(x)                 # (B*H*W, 1)
    #     return x.reshape(B, 1, H, W)
    def forward(self, x):# with maxpooling
        B, C, H, W = x.shape
        # Squeeze spatial dims 2x before pixel-wise processing
        x = F.max_pool2d(x, kernel_size=2, stride=2)   # (B, C, H//2, W//2)
        _, _, H2, W2 = x.shape

        x = x.permute(0, 2, 3, 1).reshape(-1, C).unsqueeze(1)  # (B*H2*W2, 1, C)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)          # (B*H2*W2, out_dim)
        x = self.head(x)                 # (B*H2*W2, 1)
        x = x.reshape(B, 1, H2, W2)

        # Restore original spatial size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


class BidirectionalWaveNetEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, residual_ch: int = 32,
                 skip_ch: int = 64, T: int = 280):
        super().__init__()
        self.residual_ch = residual_ch
        self.dilations   = [1, 2, 4, 8, 16, 1, 2, 4, 8]

        self.input_conv = nn.Conv1d(in_ch, residual_ch, kernel_size=1)

        self.residual_convs = nn.ModuleList([
            nn.Conv1d(residual_ch, residual_ch * 2,
                      kernel_size=3, dilation=d, padding=d)
            for d in self.dilations
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(residual_ch, skip_ch, kernel_size=1)
            for _ in self.dilations
        ])
        self.residual_proj = nn.ModuleList([
            nn.Conv1d(residual_ch, residual_ch, kernel_size=1)
            for _ in self.dilations
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(residual_ch) for _ in self.dilations
        ])
        self.skip_se = nn.ModuleList([
            SEBlock1d(skip_ch) for _ in self.dilations
        ])

        # T here = number of channels (C), not fixed 280 anymore
        self.time_weights = nn.Parameter(torch.ones(T))
        nn.init.constant_(self.time_weights[:min(70, T)], 2.0)
        nn.init.constant_(self.time_weights[min(70, T):], 1.0)

        self.output_se   = SEBlock1d(skip_ch)
        self.output_pool = nn.AdaptiveAvgPool1d(1)
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(skip_ch, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.head = nn.Linear(64, 1)     # pixel logit
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.head.bias, -2.2)  # class-imbalance prior

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C).unsqueeze(1)  # (B*H*W, 1, C)

        x = x * self.time_weights.unsqueeze(0).unsqueeze(0)
        x = self.input_conv(x)

        skip_sum = None
        for res_conv, skip_conv, res_proj, norm, se in zip(
            self.residual_convs, self.skip_convs,
            self.residual_proj, self.norms, self.skip_se
        ):
            residual = x
            h        = res_conv(x)
            h        = torch.tanh(h[:, :self.residual_ch]) \
                     * torch.sigmoid(h[:, self.residual_ch:])

            skip     = se(skip_conv(h))
            skip_sum = skip if skip_sum is None else skip_sum + skip
            x        = norm(res_proj(h) + residual)

        skip_sum = self.output_se(F.relu(skip_sum))
        skip_sum = self.output_pool(skip_sum)
        x        = self.output_head(skip_sum)    # (B*H*W, 64)
        x        = self.head(x)                  # (B*H*W, 1)
        return x.reshape(B, 1, H, W)




class FusionWeight(nn.Module):
    """
    Single trainable scalar w that blends CNN and XGB probabilities:
        combined = w * cnn_prob + (1 - w) * xgb_prob

    w is stored as an unconstrained parameter and mapped through sigmoid
    so it always stays in (0, 1).  Initialised at sigmoid(0) = 0.5.
    """
    def __init__(self, init_logit: float = 0.0):
        super().__init__()
        self.logit_w = nn.Parameter(torch.tensor(float(init_logit)))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_w)

    def weight(self) -> float:
        """Convenience: current weight as a Python float."""
        return torch.sigmoid(self.logit_w).item()


_MODEL_REGISTRY = {
    'cnn':      RefinementCNN,
    'cnn_se':   RefinementCNNSE,
    'cnn_skip': RefinementCNNSkip,
    'mlp':      RefinementMLP,
    'tcn':      TCNEncoder,
    'wavenet':  BidirectionalWaveNetEncoder
}


def build_hybrid_model(
    mode: str,
    n_raw_ch: int,
    n_filters: int = 32,
    model_name: str = 'cnn',
) -> nn.Module:
    if mode == 'prob_only':
        in_ch = 1
    elif mode == 'prob_feat':
        in_ch = n_raw_ch + 1
    elif mode in ('parallel', 'nn_only'):
        in_ch = n_raw_ch
    else:
        raise ValueError(f"Unknown mode {mode!r}. "
                         "Choose 'prob_only', 'prob_feat', 'parallel', or 'nn_only'.")

    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name {model_name!r}. "
                         f"Choose from {list(_MODEL_REGISTRY)}.")

    cls = _MODEL_REGISTRY[model_name]
    if cls is RefinementMLP:
        return cls(in_ch)  
    elif cls is BidirectionalWaveNetEncoder:
        return cls(in_ch=1, residual_ch=n_filters, T=in_ch) 
    elif cls is TCNEncoder:
        return cls(in_ch=1, hidden=n_filters)   # already correct, no T needed        # MLP uses fixed hidden=(64,32,32), no n_filters
    return cls(in_ch, n_filters=n_filters)