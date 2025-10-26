# models/dualenc_unetpp.py
"""
Dual-Encoder UNet++:
- Encoder-A: MBNeXt (Fused-MBConv @ H, MBConv @ H/2, ConvNeXt-v2 deeper)
- Encoder-B: DenseNeXt (Dense @ H/H2, ConvNeXt-v2 deeper)
- Fusion: per-stage (H, H/2, H/4, H/8, H/16) and bottleneck
- Decoder: shared UNet++ grid with ConvNeXt-v2 fusion nodes
- Deep supervision heads at X[0][1..D], upsampled to input size.

Design choices:
- Fusion = Concat -> 1x1 reduce to target row width -> ConvNeXtV2 block.
- Project-before-upsample in Up() to save memory.
- Gradient checkpoint option on ConvNeXt stages.
"""

from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

# --------------------------- Norm helpers ---------------------------

class LN2d(nn.Module):
    """LayerNorm-like over channels via GroupNorm(1, C)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels, eps=eps, affine=True)
    def forward(self, x): return self.gn(x)

def _gn_groups(c: int, groups: int) -> int:
    for g in (groups, 16, 8, 4, 2, 1):
        if c % g == 0: return g
    return 1

def Norm2d(c: int, kind: str = "gn", gn_groups: int = 8) -> nn.Module:
    if kind == "ln": return LN2d(c)
    return nn.GroupNorm(_gn_groups(c, gn_groups), c)

# ------------------------- ConvNeXt-v2 blocks ------------------------

try:
    from torch.utils.checkpoint import checkpoint as ckpt
    _HAVE_CKPT = True
except Exception:
    _HAVE_CKPT = False

class GRN(nn.Module):
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
    """LN2d -> Conv 2x2 stride 2 (downsample + channel change)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.ln = LN2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)
    def forward(self, x):
        return self.conv(self.ln(x))

# ------------------------- MBConv family (branch A) ------------------

class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, se_ratio: float = 0.25):
        super().__init__()
        hid = max(1, int(ch * se_ratio))
        self.fc1 = nn.Conv2d(ch, hid, 1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hid, ch, 1)
        self.gate = nn.Sigmoid()
    def forward(self, x):
        s = x.mean((2,3), keepdim=True)
        s = self.fc1(s); s = self.act(s); s = self.fc2(s)
        return x * self.gate(s)

class FusedMBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, exp: int = 4, k: int = 5,
                 norm: str = "gn", gn_groups: int = 8, se_ratio: float = 0.0):
        super().__init__()
        mid = in_ch * exp
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=k, padding=k//2, bias=False)
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
    def __init__(self, in_ch: int, out_ch: int, exp: int = 6, k: int = 5,
                 norm: str = "gn", gn_groups: int = 8, se_ratio: float = 0.25):
        super().__init__()
        mid = in_ch * exp
        self.pw1  = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1  = Norm2d(mid, kind=norm, gn_groups=gn_groups)
        self.act1 = nn.SiLU(inplace=True)
        self.dw   = nn.Conv2d(mid, mid, kernel_size=k, padding=k//2, groups=mid, bias=False)
        self.bn2  = Norm2d(mid, kind=norm, gn_groups=gn_groups)
        self.act2 = nn.SiLU(inplace=True)
        self.se   = SqueezeExcite(mid, se_ratio) if se_ratio and se_ratio > 0 else nn.Identity()
        self.pw2  = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn3  = Norm2d(out_ch, kind=norm, gn_groups=gn_groups)
        self.residual = (in_ch == out_ch)
    def forward(self, x):
        s = x
        x = self.pw1(x); x = self.bn1(x); x = self.act1(x)
        x = self.dw(x);  x = self.bn2(x); x = self.act2(x)
        x = self.se(x)
        x = self.pw2(x); x = self.bn3(x)
        return x + s if self.residual else x

# -------------------------- Dense blocks (branch B) ------------------

class DenseLayer(nn.Sequential):
    def __init__(self, in_ch: int, growth: int, drop_p: float = 0.2):
        super().__init__()
        self.add_module("bn",   nn.BatchNorm2d(in_ch))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(in_ch, growth, kernel_size=3, padding=1, bias=True))
        self.add_module("drop", nn.Dropout2d(drop_p))

class DenseBlock(nn.Module):
    """Classic DenseNet-style block. Exposes .out_ch."""
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

# --------------------------- Shared decoder utils --------------------

class Up(nn.Module):
    """Project BEFORE upsampling to save memory."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
    def forward(self, x, size_hw: Tuple[int, int]):
        x = self.proj(x)
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

class FuseConvNeXt(nn.Module):
    """Concat -> 1x1 reduce to out_ch -> ConvNeXtV2 block."""
    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.blk    = ConvNeXtV2Block(out_ch, drop_path)
    def forward(self, *feats):
        x = torch.cat(feats, dim=1)
        x = self.reduce(x)
        return self.blk(x)

class FuseTwo(nn.Module):
    """Fuse two tensors with possibly different channels: concat -> 1x1 -> ConvNeXtV2."""
    def __init__(self, in_ch1: int, in_ch2: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch1 + in_ch2, out_ch, kernel_size=1, bias=True)
        self.blk    = ConvNeXtV2Block(out_ch, drop_path)
    def forward(self, a, b):
        x = torch.cat([a, b], dim=1)
        x = self.reduce(x)
        return self.blk(x)

# --------------------------- Encoder branches ------------------------

class MBNeXtEncoder(nn.Module):
    """
    Stage0: Fused-MBConv (H)
    Stage1: MBConv (+SE) (H/2)
    then ConvNeXt-v2 at H/4, H/8, H/16 + bottleneck at H/32
    """
    def __init__(self,
                 in_ch: int = 3,
                 stage0_out: int = 96,
                 stage1_out: int = 160,
                 stage0_blocks: int = 3,
                 stage1_blocks: int = 4,
                 stage0_exp: int = 4,
                 stage1_exp: int = 6,
                 stage0_k:   int = 5,
                 stage1_k:   int = 5,
                 stage0_se:  float = 0.0,
                 stage1_se:  float = 0.25,
                 norm_mb:    str   = "gn",
                 gn_groups:  int   = 8,
                 dims_cnx:   Tuple[int,int,int,int] = (192, 288, 384, 480),
                 cnx_depths: Tuple[int,int,int] = (3,3,3),
                 bot_depth:  int = 3,
                 drop_path:  float = 0.0,
                 use_checkpoint: bool = True):
        super().__init__()

        self.stem = nn.Conv2d(in_ch, stage0_out, kernel_size=3, padding=1, bias=True)

        b0 = []; ch = stage0_out
        for _ in range(stage0_blocks):
            b0.append(FusedMBConv(ch, stage0_out, exp=stage0_exp, k=stage0_k,
                                  norm=norm_mb, gn_groups=gn_groups, se_ratio=stage0_se))
            ch = stage0_out
        self.enc0 = nn.Sequential(*b0)
        self.pool0 = nn.MaxPool2d(2)

        b1 = []; ch = stage0_out
        for _ in range(stage1_blocks):
            b1.append(MBConv(ch, stage1_out, exp=stage1_exp, k=stage1_k,
                             norm=norm_mb, gn_groups=gn_groups, se_ratio=stage1_se))
            ch = stage1_out
        self.enc1 = nn.Sequential(*b1)

        self.down12 = DownsampleCNX(stage1_out, dims_cnx[0])
        self.enc2   = ConvNeXtStage(dims_cnx[0], cnx_depths[0], drop_path, checkpoint=use_checkpoint)

        self.down23 = DownsampleCNX(dims_cnx[0], dims_cnx[1])
        self.enc3   = ConvNeXtStage(dims_cnx[1], cnx_depths[1], drop_path, checkpoint=use_checkpoint)

        self.down34 = DownsampleCNX(dims_cnx[1], dims_cnx[2])
        self.enc4   = ConvNeXtStage(dims_cnx[2], cnx_depths[2], drop_path, checkpoint=use_checkpoint)

        self.down4b = DownsampleCNX(dims_cnx[2], dims_cnx[3])
        self.bot    = ConvNeXtStage(dims_cnx[3], bot_depth, drop_path, checkpoint=use_checkpoint)

        self.c0 = stage0_out; self.c1 = stage1_out
        self.d0, self.d1, self.d2, self.d3 = dims_cnx

    def forward(self, x):
        x  = self.stem(x)
        e0 = self.enc0(x)              # [B,c0,H,W]
        p0 = self.pool0(e0)            # [B,c0,H/2,W/2]
        e1 = self.enc1(p0)             # [B,c1,H/2,W/2]

        d12 = self.down12(e1); e2 = self.enc2(d12)   # [B,d0,H/4,W/4]
        d23 = self.down23(e2); e3 = self.enc3(d23)   # [B,d1,H/8,W/8]
        d34 = self.down34(e3); e4 = self.enc4(d34)   # [B,d2,H/16,W/16]

        db  = self.down4b(e4);  bot = self.bot(db)   # [B,d3,H/32,W/32]
        return e0, e1, e2, e3, e4, bot


class DenseNeXtEncoder(nn.Module):
    """
    Stage0/1: Dense blocks at H / H/2, then ConvNeXt-v2 deeper like MBNeXt.
    """
    def __init__(self,
                 in_ch: int = 3,
                 stem_out: int = 48,
                 growth: int = 16,
                 dense_layers: Tuple[int,int] = (3,4),
                 dims_cnx:   Tuple[int,int,int,int] = (192, 288, 384, 480),
                 cnx_depths: Tuple[int,int,int] = (3,3,3),
                 bot_depth:  int = 3,
                 drop_path:  float = 0.0,
                 use_checkpoint: bool = True):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, stem_out, kernel_size=3, padding=1, bias=True)

        self.enc0 = DenseBlock(stem_out, growth, dense_layers[0], drop_p=0.2); c0 = self.enc0.out_ch
        self.pool0 = nn.MaxPool2d(2)
        self.enc1 = DenseBlock(c0, growth, dense_layers[1], drop_p=0.2); c1 = self.enc1.out_ch

        self.down12 = DownsampleCNX(c1, dims_cnx[0])
        self.enc2   = ConvNeXtStage(dims_cnx[0], cnx_depths[0], drop_path, checkpoint=use_checkpoint)

        self.down23 = DownsampleCNX(dims_cnx[0], dims_cnx[1])
        self.enc3   = ConvNeXtStage(dims_cnx[1], cnx_depths[1], drop_path, checkpoint=use_checkpoint)

        self.down34 = DownsampleCNX(dims_cnx[1], dims_cnx[2])
        self.enc4   = ConvNeXtStage(dims_cnx[2], cnx_depths[2], drop_path, checkpoint=use_checkpoint)

        self.down4b = DownsampleCNX(dims_cnx[2], dims_cnx[3])
        self.bot    = ConvNeXtStage(dims_cnx[3], bot_depth, drop_path, checkpoint=use_checkpoint)

        self.c0 = self.enc0.out_ch; self.c1 = self.enc1.out_ch
        self.d0, self.d1, self.d2, self.d3 = dims_cnx

    def forward(self, x):
        x  = self.stem(x)
        e0 = self.enc0(x)
        p0 = self.pool0(e0)
        e1 = self.enc1(p0)

        d12 = self.down12(e1); e2 = self.enc2(d12)
        d23 = self.down23(e2); e3 = self.enc3(d23)
        d34 = self.down34(e3); e4 = self.enc4(d34)

        db  = self.down4b(e4); bot = self.bot(db)
        return e0, e1, e2, e3, e4, bot

# ------------------------- Dual-Encoder UNet++ -----------------------

class DualEncUNetPP(nn.Module):
    """
    Two parallel encoders (MBNeXt + DenseNeXt). Their feature maps at each
    encoder level and their bottlenecks are fused, then a single UNet++
    decoder operates on the fused stream.

    Args:
      unify_channels: (c0, c1, d0, d1, d2, d3) target widths after fusion
      dec_depth: UNet++ columns (=1). deep_supervision=True -> list of logits

    Memory tips:
      - project-before-upsample in Up()
      - set dec_depth=2 if VRAM is tight; use checkpointing in encoders
    """
    def __init__(self,
                 n_classes: int = 2,
                 in_channels: int = 3,
                 # Branch A (MBNeXt) sizing
                 a_stage0_out: int = 96,
                 a_stage1_out: int = 160,
                 a_stage0_blocks: int = 3,
                 a_stage1_blocks: int = 4,
                 a_stage0_exp: int = 4,
                 a_stage1_exp: int = 6,
                 a_stage0_k:   int = 5,
                 a_stage1_k:   int = 5,
                 a_stage0_se:  float = 0.0,
                 a_stage1_se:  float = 0.25,
                 a_norm_mb:    str   = "gn",
                 a_gn_groups:  int   = 8,
                 a_dims_cnx:   Tuple[int,int,int,int] = (192, 288, 384, 480),
                 a_cnx_depths: Tuple[int,int,int] = (3,3,3),
                 a_bot_depth:  int = 3,
                 # Branch B (DenseNeXt) sizing
                 b_stem_out:   int   = 48,
                 b_growth:     int   = 16,
                 b_dense_layers: Tuple[int,int] = (3,4),
                 b_dims_cnx:   Tuple[int,int,int,int] = (192, 288, 384, 480),
                 b_cnx_depths: Tuple[int,int,int] = (3,3,3),
                 b_bot_depth:  int = 3,
                 # Shared
                 drop_path:    float = 0.0,
                 use_checkpoint: bool = True,
                 unify_channels: Tuple[int,int,int,int,int,int] = (96, 160, 224, 320, 384, 480),
                 dec_depth:    int = 2,
                 deep_supervision: bool = True):
        super().__init__()
        assert dec_depth >= 1, "UNet++ dec_depth must be >= 1"
        self.dec_depth = dec_depth
        self.deep_supervision = deep_supervision
        self.n_classes = n_classes

        # Encoders
        self.encA = MBNeXtEncoder(
            in_ch=in_channels,
            stage0_out=a_stage0_out, stage1_out=a_stage1_out,
            stage0_blocks=a_stage0_blocks, stage1_blocks=a_stage1_blocks,
            stage0_exp=a_stage0_exp, stage1_exp=a_stage1_exp,
            stage0_k=a_stage0_k, stage1_k=a_stage1_k,
            stage0_se=a_stage0_se, stage1_se=a_stage1_se,
            norm_mb=a_norm_mb, gn_groups=a_gn_groups,
            dims_cnx=a_dims_cnx, cnx_depths=a_cnx_depths,
            bot_depth=a_bot_depth, drop_path=drop_path,
            use_checkpoint=use_checkpoint
        )
        self.encB = DenseNeXtEncoder(
            in_ch=in_channels,
            stem_out=b_stem_out, growth=b_growth,
            dense_layers=b_dense_layers,
            dims_cnx=b_dims_cnx, cnx_depths=b_cnx_depths,
            bot_depth=b_bot_depth, drop_path=drop_path,
            use_checkpoint=use_checkpoint
        )

        # Target fused widths (row channels + bottleneck)
        c0, c1, d0, d1, d2, d3 = unify_channels
        self.row_channels = [c0, c1, d0, d1, d2]  # rows i=0..4

        # Per-level fusion (A,B) -> target width
        self.fuse0 = FuseTwo(self.encA.c0, self.encB.c0, c0, drop_path)
        self.fuse1 = FuseTwo(self.encA.c1, self.encB.c1, c1, drop_path)
        self.fuse2 = FuseTwo(self.encA.d0, self.encB.d0, d0, drop_path)
        self.fuse3 = FuseTwo(self.encA.d1, self.encB.d1, d1, drop_path)
        self.fuse4 = FuseTwo(self.encA.d2, self.encB.d2, d2, drop_path)
        self.fuseB = FuseTwo(self.encA.d3, self.encB.d3, d3, drop_path)  # bottleneck

        # Enhance row-4 (H/16) with upsampled fused bottleneck
        self.up_b4  = Up(d3, d2)
        self.fuse4e = FuseConvNeXt(d2 + d2, d2, drop_path)

        # UNet++ grid wiring
        self.up_nodes  = nn.ModuleDict()
        self.fuse_nodes= nn.ModuleDict()
        L = 5
        for j in range(1, dec_depth + 1):
            for i in range(0, L - j):
                in_up_ch   = self.row_channels[i+1]
                out_row_ch = self.row_channels[i]
                self.up_nodes[f"up_{i+1}_{j-1}"] = Up(in_up_ch, out_row_ch)
                in_fuse_ch = out_row_ch * (1 + j)  # up + j skips
                self.fuse_nodes[f"fuse_{i}_{j}"] = FuseConvNeXt(in_fuse_ch, out_row_ch, drop_path)

        # Heads (deep supervision)
        self.heads = nn.ModuleList([nn.Conv2d(self.row_channels[0], n_classes, 1, bias=True)
                                    for _ in range(dec_depth)])

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Branch A (MBNeXt)
        a0, a1, a2, a3, a4, ab = self.encA(x)
        # Branch B (DenseNeXt)
        b0, b1, b2, b3, b4, bb = self.encB(x)

        # Fuse per-level
        f0 = self.fuse0(a0, b0)   # [B,c0,H,W]
        f1 = self.fuse1(a1, b1)   # [B,c1,H/2,W/2]
        f2 = self.fuse2(a2, b2)   # [B,d0,H/4,W/4]
        f3 = self.fuse3(a3, b3)   # [B,d1,H/8,W/8]
        f4 = self.fuse4(a4, b4)   # [B,d2,H/16,W/16]

        fb = self.fuseB(ab, bb)   # [B,d3,H/32,W/32]
        u4 = self.up_b4(fb, (f4.shape[-2], f4.shape[-1]))  # d3->d2 at H/16
        f4e = self.fuse4e(u4, f4)                          # enhanced row-4 feature

        # UNet++ grid X[i][j]
        X: Dict[int, Dict[int, torch.Tensor]] = {i: {} for i in range(5)}
        X[0][0], X[1][0], X[2][0], X[3][0], X[4][0] = f0, f1, f2, f3, f4e

        for j in range(1, self.dec_depth + 1):
            for i in range(0, 5 - j):
                up_in = X[i + 1][j - 1]  # always from node below in previous column
                in_hw = X[i][0].shape[-2:]
                up = self.up_nodes[f"up_{i+1}_{j-1}"](up_in, in_hw)

                feats = [up] + [X[i][k] for k in range(0, j)]
                fuse = self.fuse_nodes[f"fuse_{i}_{j}"]
                if self.training:
                    X[i][j] = ckpt(self.fuse_nodes[f"fuse_{i}_{j}"], *feats, use_reentrant=False)
                else:
                    X[i][j] = fuse(*feats)

        # Deep supervision heads on X[0][1..D]
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
