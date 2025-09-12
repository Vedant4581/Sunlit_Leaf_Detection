# models/densenext_unet.py
# DenseNeXt-UNet: early Dense blocks + deep ConvNeXt-v2 stages + UNet decoder
# Works with your existing train_sunlit.py (after adding selection logic below).

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- building blocks -----------------------------

def LN2d(c: int) -> nn.Module:
    """LayerNorm for 2D feature maps; GroupNorm(1, C) is a good proxy."""
    return nn.GroupNorm(1, c)

class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt v2)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 norm over spatial dims, per-channel
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x

class DropPath(nn.Module):
    """Stochastic depth. No-op if drop_prob=0 or not training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask

class ConvNeXtV2Block(nn.Module):
    """
    DWConv(7x7) -> LN2d -> PW 1x1 (expand 4x) -> GELU -> GRN -> PW 1x1 (project) + residual
    """
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.ln = LN2d(dim)
        self.pw1 = nn.Conv2d(dim, 4*dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.grn = GRN(4*dim)
        self.pw2 = nn.Conv2d(4*dim, dim, kernel_size=1, bias=True)
        self.drop = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        x = self.drop(x)
        return x + residual

class ConvNeXtStage(nn.Module):
    def __init__(self, dim: int, depth: int, drop_path: float = 0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(dim, drop_path) for _ in range(depth)])

    def forward(self, x): return self.blocks(x)

class DownsampleCNX(nn.Module):
    """ConvNeXt-style downsample: LN2d + 2x2 stride-2 conv to change C and H,W."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            LN2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True),
        )
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    """Bilinear upsample + 1x1 projection to target channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
    def forward(self, x, size_hw: Tuple[int, int]):
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.proj(x)

class FuseConvNeXt(nn.Module):
    """Fuse concat(skip, up) -> 1x1 reduce -> ConvNeXtV2Block."""
    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.block  = ConvNeXtV2Block(out_ch, drop_path)
    def forward(self, x_up: torch.Tensor, skip: torch.Tensor):
        # align spatial sizes robustly
        H, W = skip.shape[-2], skip.shape[-1]
        if x_up.shape[-2:] != (H, W):
            x_up = F.interpolate(x_up, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([x_up, skip], dim=1)
        x = self.reduce(x)
        x = self.block(x)
        return x

# ----------------------------- Dense block (early) -----------------------------

class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int, drop_p: float = 0.2):
        super().__init__()
        self.bn   = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, growth, kernel_size=3, padding=1, bias=True)
        self.drop = nn.Dropout2d(drop_p) if drop_p and drop_p > 0 else nn.Identity()
        self.growth = growth
    def forward(self, x):
        y = self.conv(self.relu(self.bn(x)))
        y = self.drop(y)
        return torch.cat([x, y], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_ch: int, growth: int, n_layers: int, drop_p: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([DenseLayer(in_ch + i*growth, growth, drop_p) for i in range(n_layers)])
        self.out_ch = in_ch + n_layers * growth
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --------------------------------- The model ----------------------------------

class DenseNeXtUNet(nn.Module):
    """
    Hybrid encoder:
      - Stage 0/1: Dense blocks (local detail)
      - Stage 2/3/4 (+ bottleneck): ConvNeXt-v2 stages (context, stability)

    Decoder: UNet-style with ConvNeXt-v2 fusion blocks.
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
                 drop_path: float = 0.0):
        super().__init__()
        assert len(dense_layers) == 2, "Use two early Dense stages (0 and 1)."
        assert len(cnx_depths) == 3, "Use three ConvNeXt stages (2,3,4)."

        # Stem
        self.stem = nn.Conv2d(in_channels, stem_out, kernel_size=3, padding=1, bias=True)

        # Encoder stage 0 (Dense)
        self.enc0 = DenseBlock(stem_out, growth, dense_layers[0], drop_p=0.2)   # -> c0
        c0 = self.enc0.out_ch
        self.pool0 = nn.MaxPool2d(2)

        # Encoder stage 1 (Dense)
        self.enc1 = DenseBlock(c0, growth, dense_layers[1], drop_p=0.2)         # -> c1
        c1 = self.enc1.out_ch

        # Downsample to ConvNeXt dims
        self.down12 = DownsampleCNX(c1, dims_cnx[0])   # -> H/4, dims_cnx[0]

        # Encoder stage 2 (ConvNeXt)
        self.enc2 = ConvNeXtStage(dims_cnx[0], cnx_depths[0], drop_path)

        # Downsample
        self.down23 = DownsampleCNX(dims_cnx[0], dims_cnx[1])

        # Encoder stage 3 (ConvNeXt)
        self.enc3 = ConvNeXtStage(dims_cnx[1], cnx_depths[1], drop_path)

        # Downsample
        self.down34 = DownsampleCNX(dims_cnx[1], dims_cnx[2])

        # Encoder stage 4 (ConvNeXt)
        self.enc4 = ConvNeXtStage(dims_cnx[2], cnx_depths[2], drop_path)

        # Downsample to bottleneck
        self.down4b = DownsampleCNX(dims_cnx[2], dims_cnx[3])

        # Bottleneck (ConvNeXt)
        self.bot = ConvNeXtStage(dims_cnx[3], bot_depth, drop_path)

        # Decoder (UNet-style)
        self.up_b4  = Up(dims_cnx[3], dims_cnx[2])        # 640 -> 512
        self.fuse4  = FuseConvNeXt(dims_cnx[2] + dims_cnx[2], dims_cnx[2], drop_path)

        self.up_43  = Up(dims_cnx[2], dims_cnx[1])        # 512 -> 384
        self.fuse3  = FuseConvNeXt(dims_cnx[1] + dims_cnx[1], dims_cnx[1], drop_path)

        self.up_32  = Up(dims_cnx[1], dims_cnx[0])        # 384 -> 256
        self.fuse2  = FuseConvNeXt(dims_cnx[0] + dims_cnx[0], dims_cnx[0], drop_path)

        self.up_21  = Up(dims_cnx[0], c1)                 # 256 -> c1
        self.fuse1  = FuseConvNeXt(c1 + c1, c1, drop_path)

        self.up_10  = Up(c1, c0)                          # c1 -> c0
        self.fuse0  = FuseConvNeXt(c0 + c0, c0, drop_path)

        # Head
        self.head = nn.Conv2d(c0, n_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)                 # [B,48,H,W]

        # Encoder
        e0 = self.enc0(x)                # [B,c0,H,W]
        p0 = self.pool0(e0)              # [B,c0,H/2,W/2]

        e1 = self.enc1(p0)               # [B,c1,H/2,W/2]
        d12 = self.down12(e1)            # [B,d0,H/4,W/4]
        e2 = self.enc2(d12)              # [B,d0,H/4,W/4]

        d23 = self.down23(e2)            # [B,d1,H/8,W/8]
        e3 = self.enc3(d23)              # [B,d1,H/8,W/8]

        d34 = self.down34(e3)            # [B,d2,H/16,W/16]
        e4 = self.enc4(d34)              # [B,d2,H/16,W/16]

        db  = self.down4b(e4)            # [B,d3,H/32,W/32]
        bot = self.bot(db)               # [B,d3,H/32,W/32]

        # Decoder
        u4  = self.up_b4(bot, (e4.shape[-2], e4.shape[-1]))
        f4  = self.fuse4(u4, e4)

        u3  = self.up_43(f4, (e3.shape[-2], e3.shape[-1]))
        f3  = self.fuse3(u3, e3)

        u2  = self.up_32(f3, (e2.shape[-2], e2.shape[-1]))
        f2  = self.fuse2(u2, e2)

        u1  = self.up_21(f2, (e1.shape[-2], e1.shape[-1]))
        f1  = self.fuse1(u1, e1)

        u0  = self.up_10(f1, (e0.shape[-2], e0.shape[-1]))
        f0  = self.fuse0(u0, e0)

        logits = self.head(f0)
        return logits
