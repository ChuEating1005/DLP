from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(denormalize(images), nrow=nrow)
    save_image(grid, str(path))


def save_individual_images(
    images: torch.Tensor,
    out_dir: str | Path,
    name_fmt: str = "{idx:03d}.png",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = denormalize(images)
    for i, img in enumerate(images):
        save_image(img, str(out_dir / name_fmt.format(idx=i)))


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
