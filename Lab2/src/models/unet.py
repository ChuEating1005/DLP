import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Original U-Net (Ronneberger et al., 2015) — faithful implementation
#
# Key differences from "modern" UNet:
#   1. All 3x3 convolutions are VALID (no padding) — spatial dims shrink by 2
#      per conv layer.
#   2. NO BatchNorm — the original paper uses only Conv -> ReLU.
#   3. Skip connections use CENTER-CROP of encoder features to match the
#      (smaller) decoder features before concatenation.
#   4. The output is smaller than the input.  For a 572x572 input the output
#      is 388x388.
#
# To obtain a segmentation map that matches the target mask size we apply
# MIRROR PADDING to the input so that the valid-conv output exactly equals
# the desired spatial resolution.
#
# Mirror-padding math (all achievable outputs are 4 + 16k for k >= 0):
#   target 260 -> need input 444 -> mirror pad 92 px per side
#   target 388 -> need input 572 -> mirror pad 92 px per side
#   (The pad is always 92 regardless of target, because the total shrinkage
#    through the network is fixed at 184 pixels = 92 per side.)
# ---------------------------------------------------------------------------

# Constant: total spatial shrinkage of the network (per side)
_PAD = 92


def center_crop(x, target):
    """Center-crop *x* to match the spatial size of *target*."""
    _, _, h, w = x.shape
    _, _, th, tw = target.shape
    dh = (h - th) // 2
    dw = (w - tw) // 2
    return x[:, :, dh : dh + th, dw : dw + tw]


class DoubleConv(nn.Module):
    """(Conv3x3 valid -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.level0 = DoubleConv(in_channels, 64)
        self.level1 = DoubleConv(64, 128)
        self.level2 = DoubleConv(128, 256)
        self.level3 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e0 = self.level0(x)
        e1 = self.level1(self.pool(e0))
        e2 = self.level2(self.pool(e1))
        e3 = self.level3(self.pool(e2))
        return [e0, e1, e2, e3]


class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(512, 1024)

    def forward(self, x):
        return self.double_conv(self.pool(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.level3 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.level2 = DoubleConv(512, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.level1 = DoubleConv(256, 128)

        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.level0 = DoubleConv(128, 64)

    def forward(self, x, encoder_features):
        e0, e1, e2, e3 = encoder_features

        x = self.up3(x)
        x = torch.cat([x, center_crop(e3, x)], dim=1)
        x = self.level3(x)

        x = self.up2(x)
        x = torch.cat([x, center_crop(e2, x)], dim=1)
        x = self.level2(x)

        x = self.up1(x)
        x = torch.cat([x, center_crop(e1, x)], dim=1)
        x = self.level1(x)

        x = self.up0(x)
        x = torch.cat([x, center_crop(e0, x)], dim=1)
        x = self.level0(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.pad(x, [_PAD, _PAD, _PAD, _PAD], mode="reflect")

        encoder_features = self.encoder(x)
        bottleneck_out = self.bottleneck(encoder_features[-1])
        decoded = self.decoder(bottleneck_out, encoder_features)
        output = self.output_conv(decoded)
        return output
