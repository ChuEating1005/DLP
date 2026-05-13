import json
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _build_transform(
    image_size: int = 64
) -> transforms.Compose:
    steps: list[Any] = [
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(steps)


def encode_labels(
    label_lists: Sequence[Sequence[str]],
    obj2idx: dict[str, int],
    num_classes: int = 24,
) -> torch.Tensor:
    out = torch.zeros(len(label_lists), num_classes, dtype=torch.float32)
    for i, names in enumerate(label_lists):
        for n in names:
            out[i, obj2idx[n]] = 1.0
    return out


class ICLEVRDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 64,
    ):
        self.data_root = Path(data_root)
        with open(self.data_root / "train.json") as f:
            self.entries = list(json.load(f).items())
        with open(self.data_root / "objects.json") as f:
            self.obj2idx = json.load(f)
        self.num_classes = len(self.obj2idx)
        self.transform = _build_transform(image_size)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname, names = self.entries[idx]
        image = Image.open(self.data_root / fname)
        image = self.transform(image)
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for n in names:
            label[self.obj2idx[n]] = 1.0
        return image, label


def load_test_labels(
    data_root: str | Path,
    split: str,
    num_classes: int = 24,
) -> torch.Tensor:
    data_root = Path(data_root)
    with open(data_root / f"{split}.json") as f:
        items = json.load(f)
    with open(data_root / "objects.json") as f:
        obj2idx = json.load(f)
    return encode_labels(items, obj2idx, num_classes)
