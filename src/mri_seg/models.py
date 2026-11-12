"""Small reproducible U-Net baseline and Attention U-Net (2D, 3-channel input)."""
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class AttentionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        middle = max(channels // 2, 1)
        self.g = nn.Conv2d(channels, middle, kernel_size=1, bias=False)
        self.x = nn.Conv2d(channels, middle, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(middle, 1, kernel_size=1)

    def forward(self, skip, gate):
        score = self.psi(F.relu(self.x(skip) + self.g(gate), inplace=True)).sigmoid()
        return skip * score


class DecoderBlock(nn.Module):
    def __init__(self, skip_channels, decoder_channels, out_channels, attention=False):
        super().__init__()
        self.up_reduce = nn.Conv2d(decoder_channels, skip_channels, kernel_size=1)
        self.att = AttentionGate(skip_channels) if attention else None
        self.conv = DoubleConv(skip_channels * 2, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up_reduce(x)
        if self.att is not None:
            skip = self.att(skip, x)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, attention=False):
        super().__init__()
        b = base_channels
        self.e1 = DoubleConv(in_channels, b)
        self.e2 = DoubleConv(b, b * 2)
        self.e3 = DoubleConv(b * 2, b * 4)
        self.e4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)
        self.d4 = DecoderBlock(b * 8, b * 16, b * 8, attention)
        self.d3 = DecoderBlock(b * 4, b * 8, b * 4, attention)
        self.d2 = DecoderBlock(b * 2, b * 4, b * 2, attention)
        self.d1 = DecoderBlock(b, b * 2, b, attention)
        self.out = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        y = self.bottleneck(self.pool(s4))
        y = self.d4(y, s4)
        y = self.d3(y, s3)
        y = self.d2(y, s2)
        y = self.d1(y, s1)
        return self.out(y)  # raw logits; NO sigmoid here


def build_model(name: str, base_channels: int = 16) -> nn.Module:
    if name not in ('unet', 'attention_unet'):
        raise ValueError(f"Unknown model {name!r}, expected unet/attention_unet")
    return UNet(base_channels=base_channels, attention=(name == 'attention_unet'))
