import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet34 (two 3x3 convs with skip connection).

    Structure:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+shortcut) -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut: 1x1 conv to match dimensions when channels or spatial size change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class ResNet34Encoder(nn.Module):
    """
    ResNet34 encoder (trained from scratch, no pretrained weights).

    Architecture:
        Layer 0: Conv7x7(3, 64, stride=2) -> BN -> ReLU -> MaxPool(3, stride=2, pad=1)
                 Output: 64ch, H/4
        Layer 1: 3 x ResidualBlock(64, 64)        -> 64ch,  H/4
        Layer 2: 4 x ResidualBlock(64, 128, s=2)  -> 128ch, H/8
        Layer 3: 6 x ResidualBlock(128, 256, s=2) -> 256ch, H/16
        Layer 4: 3 x ResidualBlock(256, 512, s=2) -> 512ch, H/32
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # Initial conv + pool: reduces H,W by 4x
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        # Remaining blocks keep same dimensions
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns list of feature maps for skip connections:
            [x0, x1, x2, x3, x4]

        Resolutions (for input H x W):
            x0: 64ch,  H/4   (after initial conv+pool)
            x1: 64ch,  H/4   (after layer1)
            x2: 128ch, H/8   (after layer2)
            x3: 256ch, H/16  (after layer3)
            x4: 512ch, H/32  (after layer4) -- bottleneck
        """
        x0 = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # [B, 64, H/4, W/4]
        x1 = self.layer1(x0)  # [B, 64,  H/4,  W/4]
        x2 = self.layer2(x1)  # [B, 128, H/8,  W/8]
        x3 = self.layer3(x2)  # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)  # [B, 512, H/32, W/32]
        return [x0, x1, x2, x3, x4]


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2"""

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


class UNetDecoder(nn.Module):
    """
    UNet-style decoder paired with ResNet34 encoder.

    Decoder path (using encoder skip connections):
        Up4: upsample(512)  + skip_x3(256) -> concat(768)  -> DoubleConv -> 256
        Up3: upsample(256)  + skip_x2(128) -> concat(384)  -> DoubleConv -> 128
        Up2: upsample(128)  + skip_x1(64)  -> concat(192)  -> DoubleConv -> 64
        Up1: upsample(64)   + skip_x0(64)  -> concat(128)  -> DoubleConv -> 64
    """

    def __init__(self):
        super().__init__()
        # Up4: 512 -> 256, concat with x3 (256) -> 512 total -> 256
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256 + 256, 256)

        # Up3: 256 -> 128, concat with x2 (128) -> 256 total -> 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128 + 128, 128)

        # Up2: 128 -> 64, concat with x1 (64) -> 128 total -> 64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64 + 64, 64)

        # Up1: 64 -> 64, concat with x0 (64) -> 128 total -> 64
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64 + 64, 64)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: [x0(64,H/4), x1(64,H/4), x2(128,H/8), x3(256,H/16), x4(512,H/32)]
        """
        x0, x1, x2, x3, x4 = encoder_features

        # Up4: H/32 -> H/16, concat with x3
        d4 = self.up4(x4)  # [B, 256, H/16, W/16]
        d4 = torch.cat([d4, x3], dim=1)  # [B, 512, H/16, W/16]
        d4 = self.conv4(d4)  # [B, 256, H/16, W/16]

        # Up3: H/16 -> H/8, concat with x2
        d3 = self.up3(d4)  # [B, 128, H/8, W/8]
        d3 = torch.cat([d3, x2], dim=1)  # [B, 256, H/8, W/8]
        d3 = self.conv3(d3)  # [B, 128, H/8, W/8]

        # Up2: H/8 -> H/4, concat with x1
        d2 = self.up2(d3)  # [B, 64, H/4, W/4]
        d2 = torch.cat([d2, x1], dim=1)  # [B, 128, H/4, W/4]
        d2 = self.conv2(d2)  # [B, 64, H/4, W/4]

        # Up1: H/4 -> H/2, concat with x0
        d1 = self.up1(d2)  # [B, 64, H/2, W/2]
        # x0 is H/4, d1 is H/2 -- need to upsample x0 to match
        x0_up = F.interpolate(
            x0, size=d1.shape[2:], mode="bilinear", align_corners=False
        )
        d1 = torch.cat([d1, x0_up], dim=1)  # [B, 128, H/2, W/2]
        d1 = self.conv1(d1)  # [B, 64, H/2, W/2]

        return d1


class ResNet34UNet(nn.Module):
    """
    ResNet34 (encoder) + UNet (decoder) for binary segmentation.

    Input:  [B, 3, H, W]
    Output: [B, 1, H, W]  (logits)

    The ResNet34 encoder reduces resolution by 4x in its initial conv+pool,
    so the decoder only recovers to H/2. A final bilinear upsample restores
    the original resolution.
    """

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels)
        self.decoder = UNetDecoder()
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)
        encoder_features = self.encoder(x)
        decoded = self.decoder(encoder_features)  # [B, 64, H/2, W/2]
        logits = self.output_conv(decoded)  # [B, 1, H/2, W/2]
        # Upsample to match original input resolution
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )
        return logits
