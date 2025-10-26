# models/densenext_unetpp.py
import math
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.checkpoint import checkpoint as ckpt
    _HAVE_CKPT = True
except Exception:
    _HAVE_CKPT = False


# ---------- Norm & Blocks (ConvNeXt-v2 style) ----------

class LN2d(nn.Module):
    """LayerNorm-2d via GroupNorm(1, C) for small-batch stability."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels, eps=eps, affine=True)

    def forward(self, x):  # [B,C,H,W]
        return self.gn(x)


class GRN(nn.Module):
    """
    Global Response Normalization (as in ConvNeXt-v2).
    y = x * (gamma * (||x|| / (mean(||x||)+eps)) + beta)
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # L2 norm across spatial dims
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)   # [B,C,1,1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps) # [B,1,1,1] broadcast ok
        return x * (self.gamma * nx + self.beta + 1.0)


class ConvNeXtV2Block(nn.Module):
    """
    DWConv7x7 -> LN2d -> PW (C->4C) -> GELU -> GRN -> PW (4C->C) + residual
    """
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.ln = LN2d(dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.grn = GRN(4 * dim)
        self.pw2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=True)
        self.drop_path = drop_path

    def forward(self, x):
        shortcut = x
        x = self.dw(x)
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        if self.drop_path > 0 and self.training:
            # simple drop-path (stochastic depth)
            keep = 1.0 - self.drop_path
            mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep) / keep
            x = x * mask
        return x + shortcut


class ConvNeXtStage(nn.Module):
    """A stack of ConvNeXtV2 blocks; optional gradient checkpointing."""
    def __init__(self, dim: int, depth: int, drop_path: float = 0.0, checkpoint: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([ConvNeXtV2Block(dim, drop_path) for _ in range(depth)])
        self.checkpoint = checkpoint and _HAVE_CKPT

    def forward(self, x):
        for blk in self.blocks:
            if self.training and self.checkpoint:
                x = ckpt(blk, x)
            else:
                x = blk(x)
        return x


class DownsampleCNX(nn.Module):
    """LN2d -> Conv 2x2, stride 2."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.ln = LN2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)

    def forward(self, x):
        return self.conv(self.ln(x))


class Up(nn.Module):
    """Bilinear upsample to given size then 1x1 projection."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x, size_hw: Tuple[int, int]):
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.proj(x)


class FuseConvNeXt(nn.Module):
    """
    Fusion node: concat(feats) -> 1x1 reduce (to out_ch) -> ConvNeXtV2Block
    Used for UNet++ nodes (j>=1).
    """
    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.block  = ConvNeXtV2Block(out_ch, drop_path)

    def forward(self, *feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat(feats, dim=1)
        x = self.reduce(x)
        x = self.block(x)
        return x


# ---------- Dense blocks for high-res encoder ----------

class DenseLayer(nn.Sequential):
    def __init__(self, in_ch: int, growth: int, drop_p: float = 0.2):
        super().__init__()
        self.add_module("bn", nn.BatchNorm2d(in_ch))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(in_ch, growth, kernel_size=3, padding=1, bias=True))
        self.add_module("drop", nn.Dropout2d(drop_p))

class DenseBlock(nn.Module):
    """
    Standard DenseNet block. Keeps concatenating features along channels.
    Exposes .out_ch (in_ch + n_layers*growth).
    """
    def __init__(self, in_ch: int, growth: int, n_layers: int, drop_p: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(n_layers):
            self.layers.append(DenseLayer(ch, growth, drop_p))
            ch += growth
        self.out_ch = ch

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x


# ---------- DenseNeXt-UNet++ (UNet++-style nested decoder) ----------

class DenseNeXtUNetPP(nn.Module):
    """
    Encoder:
      - Stage 0/1: Dense blocks (detail preservation at high resolution)
      - Stage 2/3/4 (+ bottleneck): ConvNeXt-v2 stages (global/context)
    Decoder:
      - UNet++ nested skip connections (X[i][j]) with ConvNeXt-v2 fusion nodes.
    Outputs:
      - If deep_supervision=True: list of logits [X[0][1], X[0][2], ..., X[0][D]] (upsampled to input size)
      - Else: single logits from X[0][D]
    """
    def __init__(
        self,
        n_classes: int,
        in_channels: int = 3,
        growth: int = 16,
        dense_layers: Tuple[int, int] = (4, 5),
        cnx_depths: Tuple[int, int, int] = (3, 3, 3),
        bot_depth: int = 4,
        stem_out: int = 48,
        dims_cnx: Tuple[int, int, int, int] = (256, 384, 512, 640),
        dec_depth: int = 3,                 # UNet++ columns: j=1..dec_depth
        drop_path: float = 0.0,
        use_checkpoint: bool = False,
        deep_supervision: bool = True,      # return multi-head logits
    ):
        super().__init__()
        assert len(dense_layers) == 2
        assert len(cnx_depths) == 3
        assert dec_depth >= 1

        self.n_classes = n_classes
        self.dec_depth = int(dec_depth)
        self.deep_supervision = bool(deep_supervision)

        # --- Stem ---
        self.stem = nn.Conv2d(in_channels, stem_out, kernel_size=3, padding=1, bias=True)

        # --- Encoder (Dense @ high-res) ---
        self.enc0 = DenseBlock(stem_out, growth, dense_layers[0], drop_p=0.2)  # -> c0
        c0 = self.enc0.out_ch
        self.pool0 = nn.MaxPool2d(2)

        self.enc1 = DenseBlock(c0, growth, dense_layers[1], drop_p=0.2)        # -> c1
        c1 = self.enc1.out_ch

        # Down to ConvNeXt dims
        self.down12 = DownsampleCNX(c1, dims_cnx[0])   # -> d0
        d0 = dims_cnx[0]

        # --- ConvNeXt stages ---
        self.enc2 = ConvNeXtStage(d0, cnx_depths[0], drop_path, checkpoint=use_checkpoint)  # -> e2 (d0)

        self.down23 = DownsampleCNX(d0, dims_cnx[1]); d1 = dims_cnx[1]
        self.enc3 = ConvNeXtStage(d1, cnx_depths[1], drop_path, checkpoint=use_checkpoint)  # -> e3 (d1)

        self.down34 = DownsampleCNX(d1, dims_cnx[2]); d2 = dims_cnx[2]
        self.enc4 = ConvNeXtStage(d2, cnx_depths[2], drop_path, checkpoint=use_checkpoint)  # -> e4 (d2)

        self.down4b = DownsampleCNX(d2, dims_cnx[3]); d3 = dims_cnx[3]
        self.bot    = ConvNeXtStage(d3, bot_depth,    drop_path, checkpoint=use_checkpoint) # -> bot (d3)

        # Per-level channels (rows i=0..4)
        self.row_channels = [c0, c1, d0, d1, d2]

        # --- UNet++ grid modules ---
        # X[i][0] are encoder outputs, no modules needed for j=0
        # For each node X[i][j] with j>=1: we have a fusion module taking [Up(X[i+1][j-1]), X[i][0],...,X[i][j-1]]
        self.fuse_nodes = nn.ModuleDict()
        self.up_nodes   = nn.ModuleDict()

        # helper to build key strings
        def key_fuse(i, j): return f"fuse_{i}_{j}"
        def key_up(i, j):   return f"up_{iplus}_{jminus}"

        # allocate UNet++ nodes for rows i=0..(L-1)=4 and columns j=1..dec_depth
        L = 5
        for j in range(1, self.dec_depth + 1):
            for i in range(0, L - j):  # valid nodes satisfy i+j <= L-1
                # up from deeper node at (i+1, j-1) ? project to row_channels[i]
                iplus, jminus = i + 1, j - 1
                in_up_ch  = self.row_channels[iplus]
                out_row_ch = self.row_channels[i]
                self.up_nodes[f"up_{iplus}_{jminus}"] = Up(in_up_ch, out_row_ch)

                # fusion node input channels:
                #   up_proj(out_row_ch) + j * row_channels[i]  (since we concat X[i][0..j-1], each out_row_ch)
                in_fuse_ch = out_row_ch * (1 + j)
                self.fuse_nodes[key_fuse(i, j)] = FuseConvNeXt(in_fuse_ch, out_row_ch, drop_path)

        # --- Heads (1x1) for deep supervision outputs from X[0][j], j=1..dec_depth ---
        self.heads = nn.ModuleList([nn.Conv2d(self.row_channels[0], n_classes, kernel_size=1, bias=True)
                                    for _ in range(self.dec_depth)])

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Stem + encoder
        x = self.stem(x)          # [B, stem_out, H, W]
        e0 = self.enc0(x)         # [B, c0, H, W]
        p0 = self.pool0(e0)       # [B, c0, H/2, W/2]

        e1 = self.enc1(p0)        # [B, c1, H/2, W/2]
        d12 = self.down12(e1)     # [B, d0, H/4, W/4]
        e2 = self.enc2(d12)       # [B, d0, H/4, W/4]

        d23 = self.down23(e2)     # [B, d1, H/8, W/8]
        e3 = self.enc3(d23)       # [B, d1, H/8, W/8]

        d34 = self.down34(e3)     # [B, d2, H/16, W/16]
        e4 = self.enc4(d34)       # [B, d2, H/16, W/16]

        db  = self.down4b(e4)     # [B, d3, H/32, W/32]
        bot = self.bot(db)        # [B, d3, H/32, W/32]

        # Initialize UNet++ grid container
        # X[i][0] are set to encoder features
        X: Dict[int, Dict[int, torch.Tensor]] = {i: {} for i in range(5)}
        X[0][0], X[1][0], X[2][0], X[3][0], X[4][0] = e0, e1, e2, e3, e4

        # We need a virtual X[5][-1] as "bot" source for up to X[4][1]
        # We'll treat bot as coming from level 5 at column 0.
        cur = { (4, 0): e4 }  # not used directly; we will upsample from deeper by hand

        # Fill columns j = 1..dec_depth
        for j in range(1, self.dec_depth + 1):
            for i in range(0, 5 - j):
                # Always upsample from the node directly below in the previous column
                up_in = X[i + 1][j - 1]
                in_hw = X[i][0].shape[-2:]
                up = self.up_nodes[f"up_{i+1}_{j-1}"](up_in, in_hw)

                feats = [up] + [X[i][k] for k in range(0, j)]
                X[i][j] = self.fuse_nodes[f"fuse_{i}_{j}"](*feats)


        # Heads
        if self.deep_supervision:
            outs: List[torch.Tensor] = []
            for j in range(1, self.dec_depth + 1):
                o = self.heads[j - 1](X[0][j])
                if o.shape[-2:] != (H, W):
                    o = F.interpolate(o, size=(H, W), mode="bilinear", align_corners=False)
                outs.append(o)
            return outs  # shallow->deep
        else:
            logits = self.heads[-1](X[0][self.dec_depth])
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            return logits
