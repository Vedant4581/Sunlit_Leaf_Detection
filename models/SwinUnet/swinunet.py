import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def _gn(c: int) -> nn.GroupNorm:
    # Stable for small batch sizes
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

class SwinUNet(nn.Module):
    """
    Swin encoder (timm) + UNet-style decoder.
    Returns 2-class logits [B,2,H,W].
    """
    def __init__(
        self,
        num_classes: int = 2,
        in_ch: int = 3,
        backbone: str = "swinv2_tiny_window8_256",  # window=8 works nicely for 512
        pretrained: bool = True,
        dec_dim: int = 128,
        image_size: int = 512,
    ):
        super().__init__()
        # Encoder with 4 feature stages (~strides 4,8,16,32)
        self.encoder = timm.create_model(
            backbone,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=in_ch,
            pretrained=pretrained,
            img_size=(image_size, image_size),
        )
        self.enc_chs = self.encoder.feature_info.channels()  # e.g. [96, 192, 384, 768]
        c1, c2, c3, c4 = self.enc_chs

        # Lateral 1x1 to unify channel dims for the decoder
        self.lat4 = nn.Conv2d(c4, dec_dim, 1)
        self.lat3 = nn.Conv2d(c3, dec_dim, 1)
        self.lat2 = nn.Conv2d(c2, dec_dim, 1)
        self.lat1 = nn.Conv2d(c1, dec_dim, 1)

        # Top-down blocks (concat skip + conv)
        self.dec43 = DoubleConv(dec_dim * 2, dec_dim)  # p4 -> p3
        self.dec32 = DoubleConv(dec_dim * 2, dec_dim)  # p3 -> p2
        self.dec21 = DoubleConv(dec_dim * 2, dec_dim)  # p2 -> p1

        # Final prediction at stride 4 (upsampled to H×W later)
        self.head = nn.Conv2d(dec_dim, num_classes, kernel_size=1)

    @staticmethod
    def _upsample(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _to_nchw(self, t: torch.Tensor, expected_c: int) -> torch.Tensor:
        # Some timm builds return NHWC (B,H,W,C). Convert to NCHW as needed.
        if t.dim() == 4 and t.shape[1] != expected_c and t.shape[-1] == expected_c:
            return t.permute(0, 3, 1, 2).contiguous()
        return t

    def forward(self, x):
        H, W = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)  # may be NCHW or NHWC

        c1, c2, c3, c4 = self.enc_chs
        f1 = self._to_nchw(f1, c1)
        f2 = self._to_nchw(f2, c2)
        f3 = self._to_nchw(f3, c3)
        f4 = self._to_nchw(f4, c4)

        # Laterals (project to dec_dim)
        p4 = self.lat4(f4)                            # [B,D,H/32,W/32]
        p3 = self.lat3(f3)                            # [B,D,H/16,W/16]
        p2 = self.lat2(f2)                            # [B,D,H/8 ,W/8 ]
        p1 = self.lat1(f1)                            # [B,D,H/4 ,W/4 ]

        # Top-down: upsample and fuse
        u4 = self._upsample(p4, p3.shape[-2:])
        d3 = self.dec43(torch.cat([u4, p3], dim=1))   # [B,D,H/16,W/16]

        u3 = self._upsample(d3, p2.shape[-2:])
        d2 = self.dec32(torch.cat([u3, p2], dim=1))   # [B,D,H/8 ,W/8 ]

        u2 = self._upsample(d2, p1.shape[-2:])
        d1 = self.dec21(torch.cat([u2, p1], dim=1))   # [B,D,H/4 ,W/4 ]

        # Predict at stride 4, then upsample to exact input size
        logits_s4 = self.head(d1)                     # [B,2,H/4,W/4]
        logits = self._upsample(logits_s4, (H, W))    # [B,2,H,W]
        return logits