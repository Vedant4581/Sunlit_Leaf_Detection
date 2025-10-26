import torch
import torch.nn as nn
import torch.nn.functional as F

def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            _gn(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            _gn(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            _gn(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=False),
            _gn(f_int),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=False),
            _gn(f_int),
        )
        # BN on 1 channel is unstable for bs=1 -> remove it
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

        self.conv1 = ConvBlock(in_channels, c1)
        self.conv2 = ConvBlock(c1, c2)
        self.conv3 = ConvBlock(c2, c3)
        self.conv4 = ConvBlock(c3, c4)
        self.conv5 = ConvBlock(c4, c5)

        self.up5     = UpConvBlock(c5, c4)
        self.att5    = AttentionBlock(f_g=c4, f_l=c4, f_int=c4//2)
        self.upconv5 = ConvBlock(c4 + c4, c4)

        self.up4     = UpConvBlock(c4, c3)
        self.att4    = AttentionBlock(f_g=c3, f_l=c3, f_int=c3//2)
        self.upconv4 = ConvBlock(c3 + c3, c3)

        self.up3     = UpConvBlock(c3, c2)
        self.att3    = AttentionBlock(f_g=c2, f_l=c2, f_int=c2//2)
        self.upconv3 = ConvBlock(c2 + c2, c2)

        self.up2     = UpConvBlock(c2, c1)
        self.att2    = AttentionBlock(f_g=c1, f_l=c1, f_int=c1//2)
        self.upconv2 = ConvBlock(c1 + c1, c1)

        # 2-class logits for CrossEntropyLoss
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x5 = self.conv5(self.maxpool(x4))

        # decoder + attention-gated skips
        d5 = self.up5(x5)
        x4g = self.att5(g=d5, x=x4)
        d5 = torch.cat([x4g, d5], dim=1)
        d5 = self.upconv5(d5)

        d4 = self.up4(d5)
        x3g = self.att4(g=d4, x=x3)
        d4 = torch.cat([x3g, d4], dim=1)
        d4 = self.upconv4(d4)

        d3 = self.up3(d4)
        x2g = self.att3(g=d3, x=x2)
        d3 = torch.cat([x2g, d3], dim=1)
        d3 = self.upconv3(d3)

        d2 = self.up2(d3)
        x1g = self.att2(g=d2, x=x1)
        d2 = torch.cat([x1g, d2], dim=1)
        d2 = self.upconv2(d2)

        logits = self.head(d2)  # [B,2,H,W]
        return logits
