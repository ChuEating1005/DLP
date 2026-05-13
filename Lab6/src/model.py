import torch.nn as nn
from diffusers import UNet2DModel


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 24,
        sample_size: int = 64,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()

        time_embed_dim = block_out_channels[0] * 4

        self.label_proj = nn.Sequential(
            nn.Linear(num_classes, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type="identity",
            resnet_time_scale_shift="scale_shift",
        )

    def forward(self, x, t, labels):
        class_emb = self.label_proj(labels.float())
        return self.unet(x, t, class_labels=class_emb).sample
