# models/densenext_unet_gated.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    from torch.utils.checkpoint import checkpoint as ckpt
    _HAVE_CKPT = True
except Exception:
    _HAVE_CKPT = False


# ------------------------- Norm & ConvNeXt-v2 blocks -------------------------

class LN2d(nn.Module):
    '''LayerNorm-2d via GroupNorm(1, C) batch-size agnostic.'''
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels, eps=eps, affine=True)
    def forward(self, x):
        return self.gn(x)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt-v2)."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)            # [B,C,1,1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)          # [B,1,1,1]
        return x * (self.gamma * nx + self.beta + 1.0)


class ConvNeXtV2Block(nn.Module):
    """DW7x7 -> LN2d -> PW4C -> GELU -> GRN -> PW -> residual."""
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dw  = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.ln  = LN2d(dim)
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
            keep = 1.0 - self.drop_path
            mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep) / keep
            x = x * mask
        return x + shortcut


class ConvNeXtStage(nn.Module):
    """Stack of ConvNeXtV2 blocks with optional gradient checkpointing."""
    def __init__(self, dim: int, depth: int, drop_path: float = 0.0, checkpoint: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([ConvNeXtV2Block(dim, drop_path) for _ in range(depth)])
        self.checkpoint = checkpoint and _HAVE_CKPT

    def forward(self, x):
        for blk in self.blocks:
            if self.training and self.checkpoint:
                # silence PyTorch 2.4 warning: explicitly set use_reentrant
                x = ckpt(blk, x, use_reentrant=False)
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
    """Bilinear upsample to target size, then 1x1 projection."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
    def forward(self, x, size_hw: Tuple[int, int]):
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.proj(x)


class FuseConvNeXt(nn.Module):
    """Concat feats -> 1x1 reduce (out_ch) -> ConvNeXtV2 block."""
    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.blk    = ConvNeXtV2Block(out_ch, drop_path)
    def forward(self, *feats):
        x = torch.cat(feats, dim=1)
        x = self.reduce(x)
        return self.blk(x)


# ------------------------- Early Dense blocks (encoder 0/1) ------------------

class DenseLayer(nn.Sequential):
    def __init__(self, in_ch: int, growth: int, drop_p: float = 0.2):
        super().__init__()
        self.add_module("bn",   nn.BatchNorm2d(in_ch))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(in_ch, growth, kernel_size=3, padding=1, bias=True))
        self.add_module("drop", nn.Dropout2d(drop_p))


class DenseBlock(nn.Module):
    """Standard DenseNet block with channel concatenation. Exposes .out_ch."""
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


# ------------------------- Gated Attention for skips -------------------------

class AttentionGate(nn.Module):
    """
    Additive attention gate (Attention U-Net style) with LN2d:
        alpha = sigmoid(psi( ReLU( Wg(g) + Wx(x) )))
        out = x * alpha
    Shapes:
        g: [B, Fg, H, W]  (gating from decoder upsample/proj)
        x: [B, Fl, H, W]  (encoder skip)
    """
    def __init__(self, Fg: int, Fl: int, Fint: int):
        super().__init__()
        self.Wg  = nn.Sequential(LN2d(Fg), nn.Conv2d(Fg,  Fint, kernel_size=1, bias=True))
        self.Wx  = nn.Sequential(LN2d(Fl), nn.Conv2d(Fl,  Fint, kernel_size=1, bias=True))
        self.psi = nn.Sequential(nn.ReLU(inplace=True),
                                 nn.Conv2d(Fint, 1, kernel_size=1, bias=True),
                                 nn.Sigmoid())

    def forward(self, g, x):
        g_ = self.Wg(g)
        x_ = self.Wx(x)
        a  = self.psi(g_ + x_)
        return x * a  # broadcast over channels


# =============================== DenseNeXt-UNet (Gated) ======================

class DenseNeXtUNetGated(nn.Module):
    """
    DenseNeXt-UNet with **gated attention** on each skip:
      gate_i(up_i, enc_i) -> gated_enc_i; Fuse(up_i, gated_enc_i)
    Encoder:
      - Dense blocks at high res (stages 0/1)
      - ConvNeXt-v2 stages deeper (2/3/4 + bottleneck)
    Decoder:
      - UNet-style with ConvNeXt-v2 fusion blocks.
    """
    def __init__(self,
                 n_classes: int,
                 in_channels: int = 3,
                 growth: int = 16,
                 dense_layers: Tuple[int, int] = (4, 5),
                 cnx_depths: Tuple[int, int, int] = (3, 3, 3),
                 bot_depth: int = 4,
                 stem_out: int = 48,
                 dims_cnx: Tuple[int, int, int, int] = (256, 384, 512, 640),
                 drop_path: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()
        assert len(dense_layers) == 2
        assert len(cnx_depths) == 3

        # ---- Stem
        self.stem = nn.Conv2d(in_channels, stem_out, kernel_size=3, padding=1, bias=True)

        # ---- Encoder (Dense at high res)
        self.enc0 = DenseBlock(stem_out, growth, dense_layers[0], drop_p=0.2); c0 = self.enc0.out_ch
        self.pool0 = nn.MaxPool2d(2)

        self.enc1 = DenseBlock(c0, growth, dense_layers[1], drop_p=0.2); c1 = self.enc1.out_ch

        # bridge to ConvNeXt dims
        self.down12 = DownsampleCNX(c1, dims_cnx[0]); d0 = dims_cnx[0]
        self.enc2   = ConvNeXtStage(d0, cnx_depths[0], drop_path, checkpoint=use_checkpoint)

        self.down23 = DownsampleCNX(d0, dims_cnx[1]); d1 = dims_cnx[1]
        self.enc3   = ConvNeXtStage(d1, cnx_depths[1], drop_path, checkpoint=use_checkpoint)

        self.down34 = DownsampleCNX(d1, dims_cnx[2]); d2 = dims_cnx[2]
        self.enc4   = ConvNeXtStage(d2, cnx_depths[2], drop_path, checkpoint=use_checkpoint)

        self.down4b = DownsampleCNX(d2, dims_cnx[3]); d3 = dims_cnx[3]
        self.bot    = ConvNeXtStage(d3, bot_depth,    drop_path, checkpoint=use_checkpoint)

        # ---- Decoder: Up + Gated Skip + Fuse
        self.up_b4 = Up(d3, d2)          # 640 -> 512
        self.gate4 = AttentionGate(d2, d2, max(d2 // 2, 32))
        self.fuse4 = FuseConvNeXt(d2 + d2, d2, drop_path)

        self.up_43 = Up(d2, d1)          # 512 -> 384
        self.gate3 = AttentionGate(d1, d1, max(d1 // 2, 32))
        self.fuse3 = FuseConvNeXt(d1 + d1, d1, drop_path)

        self.up_32 = Up(d1, d0)          # 384 -> 256
        self.gate2 = AttentionGate(d0, d0, max(d0 // 2, 32))
        self.fuse2 = FuseConvNeXt(d0 + d0, d0, drop_path)

        self.up_21 = Up(d0, c1)          # 256 -> c1
        self.gate1 = AttentionGate(c1, c1, max(c1 // 2, 32))
        self.fuse1 = FuseConvNeXt(c1 + c1, c1, drop_path)

        self.up_10 = Up(c1, c0)          # c1 -> c0
        self.gate0 = AttentionGate(c0, c0, max(c0 // 2, 32))
        self.fuse0 = FuseConvNeXt(c0 + c0, c0, drop_path)

        # ---- Head
        self.head = nn.Conv2d(c0, n_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)

        # Encoder
        e0 = self.enc0(x)                # [B,c0,H,W]
        p0 = self.pool0(e0)

        e1 = self.enc1(p0)               # [B,c1,H/2,W/2]
        d12 = self.down12(e1)            # [B,d0,H/4,W/4]
        e2 = self.enc2(d12)              # [B,d0,H/4,W/4]

        d23 = self.down23(e2)            # [B,d1,H/8,W/8]
        e3 = self.enc3(d23)              # [B,d1,H/8,W/8]

        d34 = self.down34(e3)            # [B,d2,H/16,W/16]
        e4 = self.enc4(d34)              # [B,d2,H/16,W/16]

        db  = self.down4b(e4)            # [B,d3,H/32,W/32]
        bot = self.bot(db)               # [B,d3,H/32,W/32]

        # Decoder with gated skips
        u4 = self.up_b4(bot, (e4.shape[-2], e4.shape[-1]))
        g4 = self.gate4(u4, e4)
        f4 = self.fuse4(u4, g4)

        u3 = self.up_43(f4, (e3.shape[-2], e3.shape[-1]))
        g3 = self.gate3(u3, e3)
        f3 = self.fuse3(u3, g3)

        u2 = self.up_32(f3, (e2.shape[-2], e2.shape[-1]))
        g2 = self.gate2(u2, e2)
        f2 = self.fuse2(u2, g2)

        u1 = self.up_21(f2, (e1.shape[-2], e1.shape[-1]))
        g1 = self.gate1(u1, e1)
        f1 = self.fuse1(u1, g1)

        u0 = self.up_10(f1, (e0.shape[-2], e0.shape[-1]))
        g0 = self.gate0(u0, e0)
        f0 = self.fuse0(u0, g0)

        return self.head(f0)
