import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

# Project root: parent of src/ directory (works regardless of CWD)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA_ROOT = os.path.join(_PROJECT_ROOT, "dataset")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset for binary segmentation.
    Trimap -> binary: 1->foreground(1), 2 or 3->background(0).
    """

    def __init__(self, root=None, split="train", split_dir=None, transform=None):
        self.root = root if root else _DEFAULT_DATA_ROOT
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(self.root, "oxford-iiit-pet", "images")
        self.trimaps_dir = os.path.join(
            self.root, "oxford-iiit-pet", "annotations", "trimaps"
        )
        self.split_dir = split_dir if split_dir else self.root
        self.image_names = self._load_split()

    def _load_split(self):
        split_file = os.path.join(self.split_dir, f"{self.split}.txt")
        names = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                names.append(line.split()[0])
        return names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image = Image.open(os.path.join(self.images_dir, f"{name}.jpg")).convert("RGB")
        trimap = np.array(
            Image.open(os.path.join(self.trimaps_dir, f"{name}.png")).convert("L")
        )

        # Binary mask: 1 -> foreground(1), 2 or 3 -> background(0)
        mask = (trimap == 1).astype(np.float32)

        if self.transform:
            image = T.functional.to_image(image)
            mask_tv = tv_tensors.Mask(torch.from_numpy(mask).unsqueeze(0))
            image, mask_tv = self.transform(image, mask_tv)
            return image, mask_tv.float()

        return image, torch.from_numpy(mask).unsqueeze(0)


def get_transforms(split, image_size=256):
    """
    Returns torchvision.transforms.v2 pipeline.
    Applied jointly to (image, mask) — spatial transforms affect both,
    color transforms only affect the image (via tv_tensors.Mask type).
    """
    if split == "train":
        return T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomAffine(
                    degrees=15,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                ),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


def get_dataloader(root=None, batch_size=16, image_size=256, num_workers=4):
    train_transform = get_transforms("train", image_size)
    val_transform = get_transforms("val", image_size)

    train_dataset = OxfordPetDataset(root, split="train", transform=train_transform)
    val_dataset = OxfordPetDataset(root, split="val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
