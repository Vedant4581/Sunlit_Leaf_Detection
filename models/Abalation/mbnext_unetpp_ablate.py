from __future__ import annotations
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



class LN2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels, eps=eps, affine=True)
    def forward(self, x):
        return self.gn(x)

def _gn_groups(c: int, groups: int) -> int:
    for g in (groups, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1

def Norm2d(c: int, kind: str = "gn", gn_groups: int = 8) -> nn.Module:
    if kind == "ln":
        return LN2d(c)
    return nn.GroupNorm(_gn_groups(c, gn_groups), c)



class GRN(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, dim: int, depth: int, drop_path: float = 0.0, checkpoint: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([ConvNeXtV2Block(dim, drop_path) for _ in range(depth)])
        self.checkpoint = checkpoint and _HAVE_CKPT
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.training and self.checkpoint:
                x = ckpt(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

class DownsampleCNX(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.ln = LN2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.ln(x))

# ============================= MBConv family =============================

class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(ch * se_ratio))
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, ch, 1)
        self.gate = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean((2,3), keepdim=True)
        s = self.fc1(s); s = self.act(s); s = self.fc2(s)
        return x * self.gate(s)

class FusedMBConv(nn.Module):
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x
        x = self.conv1(x); x = self.bn1(x); x = self.act(x)
        x = self.se(x)
        x = self.conv2(x); x = self.bn2(x)
        return x + s if self.residual else x

class MBConv(nn.Module):
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def forward(self, x: torch.Tensor, size_hw: Tuple[int,int]) -> torch.Tensor:
        x = self.proj(x)
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None, groups: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class FuseConvNeXt(nn.Module):
    """Concat -> 1x1 reduce -> ConvNeXtV2 block."""
    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.blk    = ConvNeXtV2Block(out_ch, drop_path)
    def forward(self, *feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat(feats, dim=1)
        x = self.reduce(x)
        return self.blk(x)

class FusePlain(nn.Module):
    """Concat -> 1x1 reduce -> 2x(3x3 Conv-BN-SiLU)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn0    = nn.BatchNorm2d(out_ch)
        self.act0   = nn.SiLU(inplace=True)
        self.conv1  = ConvBNAct(out_ch, out_ch, k=3)
        self.conv2  = ConvBNAct(out_ch, out_ch, k=3)
    def forward(self, *feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat(feats, dim=1)
        x = self.act0(self.bn0(self.reduce(x)))
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# ============================= MBNeXt-UNet++ =============================

class MBNeXtUNetPP(nn.Module):
    def __init__(self,
                 n_classes: int,
                 in_channels: int = 3,
                 # Shallow MBConv sizing
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
                 # ConvNeXt deep encoder sizing
                 dims_cnx:   Tuple[int,int,int,int] = (224, 320, 416, 544),
                 cnx_depths: Tuple[int,int,int] = (3, 3, 3),
                 bot_depth:  int   = 3,
                 drop_path:  float = 0.0,
                 use_checkpoint: bool = True,
                 # UNet++ decoder
                 dec_depth:  int = 3,
                 deep_supervision: bool = True,
                 # Ablation toggles
                 enable_A: bool = True,
                 enable_B: bool = True):
        super().__init__()
        assert dec_depth >= 1, "dec_depth must be >= 1"
        self.n_classes = n_classes
        self.dec_depth = dec_depth
        self.deep_supervision = deep_supervision
        self.enable_A = enable_A
        self.enable_B = enable_B

        # ---- Stem
        self.stem = nn.Conv2d(in_channels, stage0_out, kernel_size=3, padding=1, bias=True)

        # ---- Stage 0: Fused-MBConv @ (H,W)
        blocks0 = []
        in_ch = stage0_out
        for _ in range(stage0_blocks):
            out_ch = stage0_out
            blocks0.append(FusedMBConv(in_ch, out_ch, exp=stage0_exp, k=stage0_k,
                                       norm=norm_mb, gn_groups=gn_groups, se_ratio=stage0_se))
            in_ch = out_ch
        self.enc0 = nn.Sequential(*blocks0)
        c0 = stage0_out
        self.pool0 = nn.MaxPool2d(2)

        # ---- Stage 1: MBConv(+SE) @ (H/2)
        blocks1 = []
        in_ch = c0
        for _ in range(stage1_blocks):
            out_ch = stage1_out
            blocks1.append(MBConv(in_ch, out_ch, exp=stage1_exp, k=stage1_k,
                                  norm=norm_mb, gn_groups=gn_groups, se_ratio=stage1_se))
            in_ch = out_ch
        self.enc1 = nn.Sequential(*blocks1)
        c1 = stage1_out

        # ---- Bridge to deep pyramid (A on/off)
        if self.enable_A:
            self.down12 = DownsampleCNX(c1, dims_cnx[0]); d0 = dims_cnx[0]
            self.enc2   = ConvNeXtStage(d0, cnx_depths[0], drop_path, checkpoint=use_checkpoint)
            self.down23 = DownsampleCNX(d0, dims_cnx[1]); d1 = dims_cnx[1]
            self.enc3   = ConvNeXtStage(d1, cnx_depths[1], drop_path, checkpoint=use_checkpoint)
            self.down34 = DownsampleCNX(d1, dims_cnx[2]); d2 = dims_cnx[2]
            self.enc4   = ConvNeXtStage(d2, cnx_depths[2], drop_path, checkpoint=use_checkpoint)
            self.down4b = DownsampleCNX(d2, dims_cnx[3]); d3 = dims_cnx[3]
            self.bot    = ConvNeXtStage(d3, bot_depth, drop_path, checkpoint=use_checkpoint)
            row_ch_2, row_ch_3, row_ch_4 = dims_cnx[0], dims_cnx[1], dims_cnx[2]
        else:
            vch = c1  # keep widths simple when A is off
            self.down12 = ConvBNAct(c1, vch, k=3, s=2)
            self.enc2   = nn.Sequential(ConvBNAct(vch, vch), ConvBNAct(vch, vch))
            self.down23 = ConvBNAct(vch, vch, k=3, s=2)
            self.enc3   = nn.Sequential(ConvBNAct(vch, vch), ConvBNAct(vch, vch))
            self.down34 = ConvBNAct(vch, vch, k=3, s=2)
            self.enc4   = nn.Sequential(ConvBNAct(vch, vch), ConvBNAct(vch, vch))
            self.down4b = ConvBNAct(vch, vch, k=3, s=2)
            self.bot    = nn.Sequential(ConvBNAct(vch, vch), ConvBNAct(vch, vch), ConvBNAct(vch, vch))
            row_ch_2 = row_ch_3 = row_ch_4 = vch

        # ---- UNet++ grid modules
        self.row_channels = [c0, c1, row_ch_2, row_ch_3, row_ch_4]
        self.fuse_nodes = nn.ModuleDict()
        self.up_nodes   = nn.ModuleDict()

        L = 5  # rows i=0..4
        for j in range(1, self.dec_depth + 1):
            for i in range(0, L - j):
                in_up_ch   = self.row_channels[i+1]
                out_row_ch = self.row_channels[i]
                self.up_nodes[f"up_{i+1}_{j-1}"] = Up(in_up_ch, out_row_ch)

                in_fuse_ch = out_row_ch * (1 + j)  # up + j skips
                if self.enable_B:
                    self.fuse_nodes[f"fuse_{i}_{j}"] = FuseConvNeXt(in_fuse_ch, out_row_ch, drop_path)
                else:
                    self.fuse_nodes[f"fuse_{i}_{j}"] = FusePlain(in_fuse_ch, out_row_ch)

        # ---- Heads for deep supervision (X[0][1..D])
        self.heads = nn.ModuleList([
            nn.Conv2d(self.row_channels[0], n_classes, kernel_size=1, bias=True)
            for _ in range(self.dec_depth)
        ])

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # Stem + shallow
        x  = self.stem(x)
        e0 = self.enc0(x)          # [B,c0,H,W]
        p0 = self.pool0(e0)        # [B,c0,H/2,W/2]
        e1 = self.enc1(p0)         # [B,c1,H/2,W/2]

        # Deep encoder
        d12 = self.down12(e1)      # [B,*,H/4,W/4]
        e2  = self.enc2(d12)
        d23 = self.down23(e2)      # [B,*,H/8,W/8]
        e3  = self.enc3(d23)
        d34 = self.down34(e3)      # [B,*,H/16,W/16]
        e4  = self.enc4(d34)
        db  = self.down4b(e4)      # [B,*,H/32,W/32]
        _   = self.bot(db)         # bottleneck computed but not concatenated in UNet++ grid

        # UNet++ grid X[i][j]
        X: Dict[int, Dict[int, torch.Tensor]] = {i: {} for i in range(5)}
        X[0][0], X[1][0], X[2][0], X[3][0], X[4][0] = e0, e1, e2, e3, e4

        for j in range(1, self.dec_depth + 1):
            for i in range(0, 5 - j):
                up_in = X[i + 1][j - 1] if not (i == 4 and j == 1) else X[4][0]
                in_hw = X[i][0].shape[-2:]
                up = self.up_nodes[f"up_{i+1}_{j-1}"](up_in, in_hw)
                feats = [up] + [X[i][k] for k in range(0, j)]
                X[i][j] = self.fuse_nodes[f"fuse_{i}_{j}"](*feats)

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

# ============================= Builder helpers =============================

def build_model_from_abc(n_classes: int,
                         A: bool, B: bool, C: bool,
                         **kwargs) -> MBNeXtUNetPP:
    dec_depth = 3 if C else 1
    model = MBNeXtUNetPP(
        n_classes=n_classes,
        dec_depth=dec_depth,
        enable_A=A,
        enable_B=B,
        deep_supervision=True,  # keep as in your training; change globally if needed
        **kwargs
    )
    return model

# Pretty banner and param counter

def count_params_millions(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def ablation_banner(A: bool, B: bool, C: bool, params_m: float) -> str:
    return f"A={int(A)} B={int(B)} C={int(C)} | params={params_m:.1f}M"

# ============================= CLI test =============================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", type=int, default=1, help="Local-Semantic Encoder on/off")
    parser.add_argument("--B", type=int, default=1, help="ConvNeXt Fusion on/off")
    parser.add_argument("--C", type=int, default=1, help="UNet++ grid (1=on, 0=UNet-like)")
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    args = parser.parse_args()

    model = build_model_from_abc(
        n_classes=args.n_classes,
        A=bool(args.A), B=bool(args.B), C=bool(args.C),
        in_channels=args.in_ch,
        # keep your preferred widths; change if you want lighter variants
        stage0_out=96, stage1_out=160,
        dims_cnx=(224,320,416,544),
        cnx_depths=(3,3,3), bot_depth=3,
        drop_path=0.0, use_checkpoint=True,
    )

    x = torch.randn(1, args.in_ch, args.H, args.W)
    model.eval()
    with torch.no_grad():
        y = model(x)
    params_m = count_params_millions(model)
    print(ablation_banner(bool(args.A), bool(args.B), bool(args.C), params_m))

    if isinstance(y, list):
        shapes = [tuple(t.shape) for t in y]
        print({f"head_{i}": s for i, s in enumerate(shapes)})
    else:
        print({"logits": tuple(y.shape)})
