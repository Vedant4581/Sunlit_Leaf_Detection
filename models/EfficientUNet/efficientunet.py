import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def _gn(c: int) -> nn.GroupNorm:
    # GroupNorm is batch-size friendly (works well with bs=1)
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)

class ConvGNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
            _gn(cout),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(cin, cout, 3, 1, 1),
            ConvGNAct(cout, cout, 3, 1, 1),
        )
    def forward(self, x): return self.block(x)

class EfficientUNet(nn.Module):
    """
    EfficientNet encoder (timm) + UNet-style decoder.
    Returns 2-class logits [B,2,H,W].
    """
    def __init__(
        self,
        num_classes: int = 2,
        in_ch: int = 3,
        backbone: str = "efficientnet_b2",  # b0/b1/b2... any timm EfficientNet
        pretrained: bool = True,
        dec_dim: int = 128,
        out_indices=(1, 2, 3, 4),          # ~strides 1/4,1/8,1/16,1/32
        freeze_encoder: bool = False,
    ):
        super().__init__()
        # Build encoder as a feature extractor
        self.encoder = timm.create_model(
            backbone,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_ch,
            pretrained=pretrained,
        )
        self.enc_chs = self.encoder.feature_info.channels()  # channels of each out
        assert len(self.enc_chs) == len(out_indices) == 4, "Expect 4 feature stages"

        c1, c2, c3, c4 = self.enc_chs

        # Project encoder features to a uniform decoder width
        self.lat4 = nn.Conv2d(c4, dec_dim, 1)
        self.lat3 = nn.Conv2d(c3, dec_dim, 1)
        self.lat2 = nn.Conv2d(c2, dec_dim, 1)
        self.lat1 = nn.Conv2d(c1, dec_dim, 1)

        # Top-down decoder (concat + convs)
        self.dec43 = DoubleConv(dec_dim * 2, dec_dim)  # p4 -> p3
        self.dec32 = DoubleConv(dec_dim * 2, dec_dim)  # p3 -> p2
        self.dec21 = DoubleConv(dec_dim * 2, dec_dim)  # p2 -> p1

        self.head = nn.Conv2d(dec_dim, num_classes, kernel_size=1)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @staticmethod
    def _upsample(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _to_nchw(self, t: torch.Tensor, expected_c: int) -> torch.Tensor:
        # timm usually returns NCHW for convnets; keep a safety guard for NHWC
        if t.dim() == 4 and t.shape[1] != expected_c and t.shape[-1] == expected_c:
            return t.permute(0, 3, 1, 2).contiguous()
        return t

    def forward(self, x):
        H, W = x.shape[-2:]

        # 4-level pyramid from timm
        f1, f2, f3, f4 = self.encoder(x)  # nominally at ~1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = self.enc_chs
        f1 = self._to_nchw(f1, c1)
        f2 = self._to_nchw(f2, c2)
        f3 = self._to_nchw(f3, c3)
        f4 = self._to_nchw(f4, c4)

        # Lateral 1×1
        p4 = self.lat4(f4)
        p3 = self.lat3(f3)
        p2 = self.lat2(f2)
        p1 = self.lat1(f1)

        # Top-down fusion
        u4 = self._upsample(p4, p3.shape[-2:])
        d3 = self.dec43(torch.cat([u4, p3], dim=1))

        u3 = self._upsample(d3, p2.shape[-2:])
        d2 = self.dec32(torch.cat([u3, p2], dim=1))

        u2 = self._upsample(d2, p1.shape[-2:])
        d1 = self.dec21(torch.cat([u2, p1], dim=1))

        # Predict at stride 4; upsample to input size
        logits_s4 = self.head(d1)
        logits = self._upsample(logits_s4, (H, W))
        return logits
