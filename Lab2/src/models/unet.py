import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    """
    Encoder (downsampling) path of UNet.

    Architecture (from the paper):
        Level 0: input  -> DoubleConv -> 64 channels
        Level 1: pool   -> DoubleConv -> 128 channels
        Level 2: pool   -> DoubleConv -> 256 channels
        Level 3: pool   -> DoubleConv -> 512 channels
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.level0 = DoubleConv(in_channels, 64)
        self.level1 = DoubleConv(64, 128)
        self.level2 = DoubleConv(128, 256)
        self.level3 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """Return list of encoder features [e0, e1, e2, e3] for skip connections."""
        e0 = self.level0(x)  # [B, 64, H, W]
        e1 = self.level1(self.pool(e0))  # [B, 128, H/2, W/2]
        e2 = self.level2(self.pool(e1))  # [B, 256, H/4, W/4]
        e3 = self.level3(self.pool(e2))  # [B, 512, H/8, W/8]
        return [e0, e1, e2, e3]


class Bottleneck(nn.Module):
    """
    Bottleneck: deepest part of UNet.
    Pool -> DoubleConv with 1024 channels.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(512, 1024)

    def forward(self, x):
        x = self.pool(x)  # [B, 512, H/16, W/16]
        x = self.double_conv(x)  # [B, 1024, H/16, W/16]
        return x


class Decoder(nn.Module):
    """
    Decoder (upsampling) path of UNet.

    At each level:
        1. Upsample (ConvTranspose2d or Upsample + Conv) to double spatial size
        2. Concatenate with corresponding encoder skip connection
        3. DoubleConv to reduce channels

    Architecture:
        Level 3: up(1024) + skip(512) -> concat(1024) -> DoubleConv -> 512
        Level 2: up(512)  + skip(256) -> concat(512)  -> DoubleConv -> 256
        Level 1: up(256)  + skip(128) -> concat(256)  -> DoubleConv -> 128
        Level 0: up(128)  + skip(64)  -> concat(128)  -> DoubleConv -> 64
    """

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
        """
        Args:
            x: bottleneck output
            encoder_features: list [e0, e1, e2, e3] from encoder (high-to-low res)
        """
        e0, e1, e2, e3 = encoder_features
        x = self.up3(x)  # [B, 512, H/8, W/8]
        x = torch.cat([x, e3], dim=1)  # [B, 1024, H/8, W/8]
        x = self.level3(x)  # [B, 512, H/8, W/8]

        x = self.up2(x)  # [B, 256, H/4, W/4]
        x = torch.cat([x, e2], dim=1)  # [B, 512, H/4, W/4]
        x = self.level2(x)  # [B, 256, H/4, W/4]

        x = self.up1(x)  # [B, 128, H/2, W/2]
        x = torch.cat([x, e1], dim=1)  # [B, 256, H/2, W/2]
        x = self.level1(x)  # [B, 128, H/2, W/2]

        x = self.up0(x)  # [B, 64, H, W]
        x = torch.cat([x, e0], dim=1)  # [B, 128, H, W]
        x = self.level0(x)  # [B, 64, H, W]

        return x


class UNet(nn.Module):
    """
    Full UNet: Encoder -> Bottleneck -> Decoder -> 1x1 Conv (output head).

    Input:  [B, 3, H, W]   (RGB image)
    Output: [B, 1, H, W]   (binary segmentation logits -- apply sigmoid at inference)
    """

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_features = self.encoder(x)
        bottleneck_out = self.bottleneck(encoder_features[-1])
        decoded = self.decoder(bottleneck_out, encoder_features)
        output = self.output_conv(decoded)
        return output
