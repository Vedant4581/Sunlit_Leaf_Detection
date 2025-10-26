# models/mbnext_unetpp.py
import math
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.checkpoint import checkpoint as ckpt
    _HAVE_CKPT = True
except Exception:
    _HAVE_CKPT = False


# ============================= Norm helpers =============================

class LN2d(nn.Module):
    """LayerNorm-like over channels via GroupNorm(1, C)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels, eps=eps, affine=True)
    def forward(self, x): return self.gn(x)

def _gn_groups(c: int, groups: int) -> int:
    # Make sure groups divides channels
    for g in (groups, 16, 8, 4, 2, 1):
        if c % g == 0: return g
    return 1

def Norm2d(c: int, kind: str = "gn", gn_groups: int = 8) -> nn.Module:
    if kind == "ln":
        return LN2d(c)
    # default GN
    return nn.GroupNorm(_gn_groups(c, gn_groups), c)


# ============================= ConvNeXt-v2 blocks =============================

class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt-v2)."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x * (self.gamma * nx + self.beta + 1.0)

class ConvNeXtV2Block(nn.Module):
    """DW7x7 -> LN2d -> PW4C -> GELU -> GRN -> PW -> residual."""
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dw  = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.ln  = LN2d(dim)
        self.pw1 = nn.Conv2d(dim, 4*dim, kernel_size=1, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.grn = GRN(4*dim)
        self.pw2 = nn.Conv2d(4*dim, dim, kernel_size=1, bias=True)
        self.drop_path = drop_path
    def forward(self, x):
        s = x
        x = self.dw(x)
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        if self.drop_path > 0 and self.training:
            keep = 1.0 - self.drop_path
            mask = torch.empty(x.size(0),1,1,1, device=x.device).bernoulli_(keep) / keep
            x = x * mask
        return x + s

class ConvNeXtStage(nn.Module):
    """Stack of ConvNeXtV2 blocks with optional gradient checkpointing."""
    def __init__(self, dim: int, depth: int, drop_path: float = 0.0, checkpoint: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([ConvNeXtV2Block(dim, drop_path) for _ in range(depth)])
        self.checkpoint = checkpoint and _HAVE_CKPT
    def forward(self, x):
        for blk in self.blocks:
            if self.training and self.checkpoint:
                x = ckpt(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

class DownsampleCNX(nn.Module):
    """LN2d -> Conv 2x2 stride 2 (channel change + spatial downsample)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.ln = LN2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)
    def forward(self, x):
        return self.conv(self.ln(x))


# ============================= MBConv family (EfficientNet/V2) =============================

class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(ch * se_ratio))
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, ch, 1)
        self.gate = nn.Sigmoid()
    def forward(self, x):
        s = x.mean((2,3), keepdim=True)
        s = self.fc1(s); s = self.act(s); s = self.fc2(s)
        return x * self.gate(s)

class FusedMBConv(nn.Module):
    """
    Fused-MBConv (EffNetV2-style)
      Conv 3x3 (in -> t*in) -> Norm -> SiLU -> Conv 1x1 (t*in -> out) -> Norm
      Residual if in==out.
    """
    def __init__(self, in_ch: int, out_ch: int, exp: int = 4, k: int = 3,
                 norm: str = "gn", gn_groups: int = 8, se_ratio: float = 0.0):
        super().__init__()
        mid = in_ch * exp
        pad = k // 2
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=k, padding=pad, bias=False)
        self.bn1   = Norm2d(mid, kind=norm, gn_groups=gn_groups)
        self.act   = nn.SiLU(inplace=True)
        self.se    = SqueezeExcite(mid, se_ratio) if se_ratio and se_ratio > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(mid, out_ch, kernel_size=1, bias=False)
        self.bn2   = Norm2d(out_ch, kind=norm, gn_groups=gn_groups)
        self.residual = (in_ch == out_ch)
    def forward(self, x):
        s = x
        x = self.conv1(x); x = self.bn1(x); x = self.act(x)
        x = self.se(x)
        x = self.conv2(x); x = self.bn2(x)
        return x + s if self.residual else x

class MBConv(nn.Module):
    """
    Classic MBConv:
      PW expand (in -> t*in) -> Norm -> SiLU
      DW kxk -> Norm -> SiLU
      SE(t*in) -> PW project (t*in -> out) -> Norm
      Residual if in==out.
    """
    def __init__(self, in_ch: int, out_ch: int, exp: int = 4, k: int = 5,
                 norm: str = "gn", gn_groups: int = 8, se_ratio: float = 0.25):
        super().__init__()
        mid = in_ch * exp
        pad = k // 2
        self.pw1  = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1  = Norm2d(mid, kind=norm, gn_groups=gn_groups)
        self.act1 = nn.SiLU(inplace=True)

        self.dw   = nn.Conv2d(mid, mid, kernel_size=k, padding=pad, groups=mid, bias=False)
        self.bn2  = Norm2d(mid, kind=norm, gn_groups=gn_groups)
        self.act2 = nn.SiLU(inplace=True)

        self.se   = SqueezeExcite(mid, se_ratio) if se_ratio and se_ratio > 0 else nn.Identity()

        self.pw2  = nn.Conv2d(mid, out_ch, kernel_size=1, bias=False)
        self.bn3  = Norm2d(out_ch, kind=norm, gn_groups=gn_groups)

        self.residual = (in_ch == out_ch)

    def forward(self, x):
        s = x
        x = self.pw1(x); x = self.bn1(x); x = self.act1(x)
        x = self.dw(x);  x = self.bn2(x); x = self.act2(x)
        x = self.se(x)
        x = self.pw2(x); x = self.bn3(x)
        return x + s if self.residual else x


# ============================= Decoder utilities =============================

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
    def forward(self, x, size_hw):
        x = self.proj(x)                               # project at low res
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)


class FuseConvNeXt(nn.Module):
    """Concat -> 1x1 reduce -> ConvNeXtV2 block."""
    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.blk    = ConvNeXtV2Block(out_ch, drop_path)
    def forward(self, *feats):
        x = torch.cat(feats, dim=1)
        x = self.reduce(x)
        return self.blk(x)


# ============================= MBNeXt-UNet++ =============================

class MBNeXtUNetPP(nn.Module):
    """
    Encoder:
      Stage-0  (H,W):   Fused-MBConv stack  -> out c0
      Pool x2
      Stage-1  (H/2):   MBConv(+SE) stack  -> out c1
      DownsampleCNX    -> d0 (H/4)
      ConvNeXt-v2 stages at H/4 (d0), H/8 (d1), H/16 (d2)
      Downsample to bottleneck d3, ConvNeXt-v2 bottleneck

    Decoder:
      UNet++ nested grid X[i][j] with ConvNeXt-v2 fusion nodes (j>=1)
      Deep supervision: heads at X[0][1..D] (upsampled to input size)
    """
    def __init__(self,
                 n_classes: int,
                 in_channels: int = 3,
                 # Shallow MBConv sizing
                 stage0_out: int = 96,          # c0 (keep compatible with your current decoder)
                 stage1_out: int = 160,         # c1
                 stage0_blocks: int = 3,
                 stage1_blocks: int = 4,
                 stage0_exp: int = 4,           # fused expansion
                 stage1_exp: int = 6,           # MBConv expansion
                 stage0_k:   int = 5,           # fused kernel (3 or 5)
                 stage1_k:   int = 5,           # MBConv kernel (3 or 5)
                 stage0_se:  float = 0.0,       # often 0.0–0.1 for fused
                 stage1_se:  float = 0.25,      # typical 0.25
                 norm_mb:    str   = "gn",      # "gn" or "ln"
                 gn_groups:  int   = 8,
                 # ConvNeXt deep encoder sizing
                 dims_cnx:   Tuple[int,int,int,int] = (224, 320, 416, 544),  # d0,d1,d2,d3
                 cnx_depths: Tuple[int,int,int] = (3, 3, 3),
                 bot_depth:  int   = 3,
                 drop_path:  float = 0.0,
                 use_checkpoint: bool = True,
                 # UNet++ decoder
                 dec_depth:  int = 3,
                 deep_supervision: bool = True):
        super().__init__()
        assert dec_depth >= 1

        self.n_classes = n_classes
        self.dec_depth = dec_depth
        self.deep_supervision = deep_supervision

        # ---- Stem
        self.stem = nn.Conv2d(in_channels, stage0_out, kernel_size=3, padding=1, bias=True)

        # ---- Stage 0: Fused-MBConv stack @ (H,W) -> c0
        blocks0 = []
        in_ch = stage0_out
        for b in range(stage0_blocks):
            out_ch = stage0_out  # keep width constant for this stage
            blocks0.append(FusedMBConv(in_ch, out_ch, exp=stage0_exp, k=stage0_k,
                                       norm=norm_mb, gn_groups=gn_groups, se_ratio=stage0_se))
            in_ch = out_ch
        self.enc0 = nn.Sequential(*blocks0)     # -> c0
        c0 = stage0_out

        self.pool0 = nn.MaxPool2d(2)

        # ---- Stage 1: MBConv(+SE) stack @ (H/2) -> c1
        blocks1 = []
        in_ch = c0
        for b in range(stage1_blocks):
            out_ch = stage1_out
            blocks1.append(MBConv(in_ch, out_ch, exp=stage1_exp, k=stage1_k,
                                  norm=norm_mb, gn_groups=gn_groups, se_ratio=stage1_se))
            in_ch = out_ch
        self.enc1 = nn.Sequential(*blocks1)     # -> c1
        c1 = stage1_out

        # ---- Bridge to ConvNeXt dims
        self.down12 = DownsampleCNX(c1, dims_cnx[0]); d0 = dims_cnx[0]

        # ---- ConvNeXt stages (H/4, H/8, H/16) + bottleneck
        self.enc2 = ConvNeXtStage(d0, cnx_depths[0], drop_path, checkpoint=use_checkpoint)  # -> e2 (d0)
        self.down23 = DownsampleCNX(d0, dims_cnx[1]); d1 = dims_cnx[1]
        self.enc3 = ConvNeXtStage(d1, cnx_depths[1], drop_path, checkpoint=use_checkpoint)  # -> e3 (d1)
        self.down34 = DownsampleCNX(d1, dims_cnx[2]); d2 = dims_cnx[2]
        self.enc4 = ConvNeXtStage(d2, cnx_depths[2], drop_path, checkpoint=use_checkpoint)  # -> e4 (d2)

        self.down4b = DownsampleCNX(d2, dims_cnx[3]); d3 = dims_cnx[3]
        self.bot    = ConvNeXtStage(d3, bot_depth, drop_path, checkpoint=use_checkpoint)

        # ---- UNet++ grid module dictionaries
        # Row channel widths: c0, c1, d0, d1, d2
        self.row_channels = [c0, c1, d0, d1, d2]
        self.fuse_nodes = nn.ModuleDict()
        self.up_nodes   = nn.ModuleDict()

        L = 5  # rows i=0..4
        for j in range(1, dec_depth + 1):
            for i in range(0, L - j):
                in_up_ch   = self.row_channels[i+1]
                out_row_ch = self.row_channels[i]
                self.up_nodes[f"up_{i+1}_{j-1}"] = Up(in_up_ch, out_row_ch)

                in_fuse_ch = out_row_ch * (1 + j)  # up + j skips
                self.fuse_nodes[f"fuse_{i}_{j}"] = FuseConvNeXt(in_fuse_ch, out_row_ch, drop_path)

        # ---- Heads for deep supervision (X[0][1..D])
        self.heads = nn.ModuleList([nn.Conv2d(self.row_channels[0], n_classes, kernel_size=1, bias=True)
                                    for _ in range(dec_depth)])

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Stem + shallow encoder (MBConv family)
        x  = self.stem(x)           # [B,c0,H,W]
        e0 = self.enc0(x)           # [B,c0,H,W]
        p0 = self.pool0(e0)         # [B,c0,H/2,W/2]

        e1 = self.enc1(p0)          # [B,c1,H/2,W/2]

        # Deep encoder (ConvNeXt)
        d12 = self.down12(e1)       # [B,d0,H/4,W/4]
        e2  = self.enc2(d12)        # [B,d0,H/4,W/4]

        d23 = self.down23(e2)       # [B,d1,H/8,W/8]
        e3  = self.enc3(d23)        # [B,d1,H/8,W/8]

        d34 = self.down34(e3)       # [B,d2,H/16,W/16]
        e4  = self.enc4(d34)        # [B,d2,H/16,W/16]

        db  = self.down4b(e4)       # [B,d3,H/32,W/32]
        bot = self.bot(db)          # [B,d3,H/32,W/32]

        # UNet++ grid X[i][j]
        X: Dict[int, Dict[int, torch.Tensor]] = {i: {} for i in range(5)}
        X[0][0], X[1][0], X[2][0], X[3][0], X[4][0] = e0, e1, e2, e3, e4

        for j in range(1, self.dec_depth + 1):
            for i in range(0, 5 - j):
                # always upsample from the node below in previous column
                up_in = X[i + 1][j - 1] if not (i == 4 and j == 1) else X[4][0]
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
            return outs
        else:
            logits = self.heads[-1](X[0][self.dec_depth])
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            return logits
