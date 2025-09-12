# models/ca_denseunetpp.py
# CA-DenseUNet++: Dense encoder + UNet++ nested decoder + Coordinate Attention
# Usage:
#   from models.ca_denseunetpp import CADenseUNetPP
#   model = CADenseUNetPP(n_classes=2, norm='bn', deep_supervision=False,
#                         ca_encoder_stages=(0,1), ca_fusion_min_j=2)

from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------ Utilities ------------------------------

def _norm2d(norm: str, c: int) -> nn.Module:
    if norm == 'bn':
        return nn.BatchNorm2d(c)
    elif norm == 'gn':
        # 8 groups is a good default for small batch sizes
        ng = 8 if c >= 8 else 1
        return nn.GroupNorm(ng, c)
    else:
        raise ValueError(f"Unknown norm '{norm}'. Use 'bn' or 'gn'.")

class ConvBNAct(nn.Module):
    """Norm -> ReLU -> Conv (3x3 by default)."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm='bn', bias=False, act=True):
        super().__init__()
        self.net = nn.Sequential(
            _norm2d(norm, in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias),
        )
        self.act = act
        self.act_fn = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.net(x)
        return self.act_fn(x) if self.act else x


# -------------------------- Coordinate Attention ------------------------

class CoordAtt(nn.Module):
    """
    Coordinate Attention (Hou et al., 2021).
    Preserves direction-aware positional info with two pooled branches.
    """
    def __init__(self, in_ch: int, reduction: int = 32, norm: str = 'bn'):
        super().__init__()
        mid = max(8, in_ch // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # (H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # (1,W)
        self.conv1  = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1    = _norm2d(norm, mid)
        self.act    = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, in_ch, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, in_ch, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [B,C,H,W]
        b, c, h, w = x.size()
        x_h = self.pool_h(x)               # [B,C,H,1]
        x_w = self.pool_w(x).transpose(2,3)  # [B,C,W,1] -> [B,C,1,W] after transpose later

        y = torch.cat([x_h, x_w], dim=2)   # concat along spatial height dimension
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2,3)           # back to [B,mid,1,W]

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


# ---------------------------- DenseNet pieces ---------------------------

class DenseLayer(nn.Module):
    """Simple (BN-ReLU-Conv3x3 + Dropout) layer, growth-ch output."""
    def __init__(self, in_ch: int, growth: int, norm: str = 'bn', drop_p: float = 0.2):
        super().__init__()
        self.norm = _norm2d(norm, in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, growth, kernel_size=3, padding=1, bias=True)
        self.drop = nn.Dropout2d(drop_p) if drop_p and drop_p > 0 else nn.Identity()

    def forward(self, x):
        y = self.conv(self.relu(self.norm(x)))
        y = self.drop(y)
        return torch.cat([x, y], dim=1)

class DenseBlock(nn.Module):
    """
    If upsample=False: returns concatenated features (input + all new layers).
    If upsample=True : returns concatenation of only new features (like Tiramisu bottleneck/up blocks).
    """
    def __init__(self, in_ch: int, growth: int, n_layers: int, upsample: bool = False,
                 norm: str = 'bn', drop_p: float = 0.2):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([
            DenseLayer(in_ch + i*growth, growth, norm=norm, drop_p=drop_p)
            for i in range(n_layers)
        ])
        self.growth = growth

    def forward(self, x):
        if self.upsample:
            new_feats = []
            for layer in self.layers:
                y = layer(x)
                # layer concatenates in forward, so new part is last growth channels
                new = y[:, -self.growth:, :, :]
                x = y
                new_feats.append(new)
            return torch.cat(new_feats, dim=1)
        else:
            for layer in self.layers:
                x = layer(x)
            return x

class TransitionDown(nn.Module):
    def __init__(self, in_ch: int, norm: str = 'bn', drop_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            _norm2d(norm, in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
            nn.Dropout2d(drop_p) if drop_p and drop_p > 0 else nn.Identity(),
            nn.MaxPool2d(2),
        )
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    """Bilinear upsample (fast & memory-friendly)."""
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, x): return self.up(x)


# -------------------------- UNet++ Fusion Node --------------------------

class FusionNode(nn.Module):
    """
    UNet++ node: concat(Up(deeper), same-level previous nodes..., encoder skip) -> Conv3x3 -> (optional CA)
    Inputs will be provided in forward(*feats) in the order:
       [Up(deeper), X^{i,0}, X^{i,1}, ..., X^{i,j-1}]
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = 'bn', use_ca: bool = True, drop_p: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, out_ch, k=3, p=1, norm=norm),
            nn.Dropout2d(drop_p) if drop_p and drop_p > 0 else nn.Identity(),
            ConvBNAct(out_ch, out_ch, k=3, p=1, norm=norm),
        )
        self.ca = CoordAtt(out_ch, reduction=32, norm=norm) if use_ca else nn.Identity()

    def forward(self, *feats):
        # Auto-align all inputs to the first tensor's spatial size (robustness to off-by-1 or stage mismatches)
        H, W = feats[0].shape[-2], feats[0].shape[-1]
        aligned = [f if f.shape[-2:] == (H, W)
                   else F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
                   for f in feats]
        x = torch.cat(aligned, dim=1)
        x = self.conv(x)
        return self.ca(x)



# -------------------------- CA-DenseUNet++ Model ------------------------

class CADenseUNetPP(nn.Module):
    """
    Dense encoder + UNet++ decoder with Coordinate Attention.

    Args:
        n_classes: number of output classes.
        in_channels: input channels (default 3).
        down_blocks: tuple of #dense layers per encoder stage.
        bottleneck_layers: #dense layers at the bottleneck (upsample=True style).
        up_blocks: tuple used to derive decoder output channels per level: dec_ch[i] = growth_rate * up_blocks[i]
                   (len(up_blocks) must equal len(down_blocks)).
        growth_rate: growth rate for dense layers.
        out_chans_first_conv: channels of initial 3x3 conv.
        norm: 'bn' or 'gn'.
        deep_supervision: if True, returns list of logits from X^{0,1},...,X^{0,D}.
        ca_encoder_stages: indices of encoder stages to apply CA after their dense block (e.g., (0,1)).
        ca_fusion_min_j: apply CA in fusion nodes only when j >= this number (e.g., 2).
        drop_p: dropout prob used inside dense layers and fusion blocks.
    """
    def __init__(self,
                 n_classes: int,
                 in_channels: int = 3,
                 down_blocks: Tuple[int, ...] = (4, 5, 7, 10, 12),
                 bottleneck_layers: int = 15,
                 up_blocks: Tuple[int, ...] = (12, 10, 7, 5, 4),
                 growth_rate: int = 16,
                 out_chans_first_conv: int = 48,
                 norm: str = 'bn',
                 deep_supervision: bool = False,
                 ca_encoder_stages: Tuple[int, ...] = (0, 1),
                 ca_fusion_min_j: int = 2,
                 drop_p: float = 0.2):
        super().__init__()

        assert len(up_blocks) == len(down_blocks), "up_blocks and down_blocks must have same length"

        self.n_classes = n_classes
        self.norm = norm
        self.deep_supervision = deep_supervision
        self.ca_encoder_stages = set(ca_encoder_stages)
        self.ca_fusion_min_j = ca_fusion_min_j

        L = len(down_blocks)               # number of encoder stages
        self.levels = L
        self.dec_depth = L - 1             # UNet++ depth

        # First conv
        self.firstconv = nn.Conv2d(in_channels, out_chans_first_conv, kernel_size=3, padding=1, bias=True)
        cur_ch = out_chans_first_conv

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.td_blocks = nn.ModuleList()
        self.ca_enc = nn.ModuleList()
        self.enc_out_ch: List[int] = []

        for i in range(L):
            n_layers = down_blocks[i]
            block = DenseBlock(cur_ch, growth_rate, n_layers, upsample=False, norm=norm, drop_p=drop_p)
            self.encoder_blocks.append(block)
            cur_ch = cur_ch + growth_rate * n_layers  # after dense concat
            self.enc_out_ch.append(cur_ch)

            # CA after dense block (high-res stages recommended, e.g., 0,1)
            if i in self.ca_encoder_stages:
                self.ca_enc.append(CoordAtt(cur_ch, reduction=32, norm=norm))
            else:
                self.ca_enc.append(nn.Identity())

            # Transition Down
            self.td_blocks.append(TransitionDown(cur_ch, norm=norm, drop_p=drop_p))

        # Bottleneck (following Tiramisu style: returns growth*bottleneck layers)
        self.bottleneck = DenseBlock(cur_ch, growth_rate, bottleneck_layers, upsample=True, norm=norm, drop_p=drop_p)
        self.bot_ch = growth_rate * bottleneck_layers
        self.ca_bot = CoordAtt(self.bot_ch, reduction=32, norm=norm)

        # Decoder target channels per level (kept constant across j at level i)
        self.dec_ch = [growth_rate * k for k in up_blocks]

        # Up sampler (shared)
        self.up = Up()

        # Build UNet++ fusion nodes
        # X[i][j] exists for i in [0, L-1-j], j in [1, dec_depth]
        self.fuse_nodes = nn.ModuleDict()
        for j in range(1, self.dec_depth + 1):
            for i in range(0, L - j):
                # Inputs: Up(X[i+1][j-1]) + X[i][0] + X[i][1] + ... + X[i][j-1]
                if j == 1:
                    # Up source channels: encoder at level i+1, except deepest uses bottleneck
                    up_src_ch = self.bot_ch if (i + 1 == L - 1) else self.enc_out_ch[i + 1]
                else:
                    up_src_ch = self.dec_ch[i + 1]
                in_ch = up_src_ch + self.enc_out_ch[i] + (j - 1) * self.dec_ch[i]
                out_ch = self.dec_ch[i]
                use_ca = (j >= self.ca_fusion_min_j)
                self.fuse_nodes[f"{i}_{j}"] = FusionNode(in_ch, out_ch, norm=norm, use_ca=use_ca, drop_p=drop_p)

        # Heads (deep supervision or single)
        if self.deep_supervision:
            self.heads = nn.ModuleList([nn.Conv2d(self.dec_ch[0], n_classes, kernel_size=1)
                                        for _ in range(1, self.dec_depth + 1)])
        else:
            self.head = nn.Conv2d(self.dec_ch[0], n_classes, kernel_size=1)

    # ------------------------------ Forward ------------------------------

    def forward(self, x):
        x = self.firstconv(x)

        # Encoder with skips
        skips: List[torch.Tensor] = []
        out = x
        for i in range(self.levels):
            out = self.encoder_blocks[i](out)      # concat inside block
            out = self.ca_enc[i](out)              # optional CA at stage i
            skips.append(out)                      # X[i][0]
            out = self.td_blocks[i](out)

        # Bottleneck + CA
        bott = self.bottleneck(out)
        bott = self.ca_bot(bott)

        # UNet++ grid X[i][j]
        X: List[List[Optional[torch.Tensor]]] = [[None for _ in range(self.dec_depth + 1)]
                                                 for _ in range(self.levels)]
        for i in range(self.levels):
            X[i][0] = skips[i]

        # Build nested decoder
        for j in range(1, self.dec_depth + 1):
            for i in range(0, self.levels - j):
                # Up source: for (i=L-2, j=1) use bottleneck; else X[i+1][j-1]
                if j == 1 and (i + 1 == self.levels - 1):
                    up_in = self.up(bott)
                else:
                    up_in = self.up(X[i + 1][j - 1])
                # Gather inputs in required order
                feats = [up_in] + [X[i][k] for k in range(0, j)]
                X[i][j] = self.fuse_nodes[f"{i}_{j}"](*feats)

        if self.deep_supervision:
            # Return list of logits from X[0][1], ..., X[0][dec_depth]
            outs = []
            for j in range(1, self.dec_depth + 1):
                outs.append(self.heads[j - 1](X[0][j]))
            return outs  # list of [B, C, H, W]
        else:
            out = self.head(X[0][self.dec_depth])
            return out  # raw logits
