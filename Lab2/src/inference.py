import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from oxford_pet import OxfordPetDataset, get_transforms
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import load_checkpoint, tta_predict, rle_encode


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_original_sizes(dataset):
    """Read original (W, H) for each image in the dataset."""
    sizes = []
    for name in dataset.image_names:
        img_path = os.path.join(dataset.images_dir, f"{name}.jpg")
        with Image.open(img_path) as img:
            sizes.append(img.size)  # (W, H)
    return sizes


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "resnet34_unet"],
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to dataset root (default: auto-detect)",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test_unet",
        choices=["test_unet", "test_res_unet"],
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Resize dimension (default: 260 for unet, 256 for resnet34_unet)",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--tta", action="store_true", help="Enable test-time augmentation"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Binarization threshold"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: submission.csv in project root)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_size is None:
        args.image_size = 260 if args.model == "unet" else 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output is None:
        args.output = os.path.join(_PROJECT_ROOT, "submission.csv")

    if args.model == "unet":
        model = UNet(in_channels=3, num_classes=1).to(device)
    else:
        model = ResNet34UNet(in_channels=3, num_classes=1).to(device)

    load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f"Checkpoint: {args.checkpoint}, threshold: {args.threshold}")

    transform = get_transforms("val", args.image_size)
    test_dataset = OxfordPetDataset(
        args.data_root, split=args.test_split, transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    original_sizes = get_original_sizes(test_dataset)

    all_probs = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            if args.tta:
                probs = tta_predict(model, images)
            else:
                probs = torch.sigmoid(model(images))
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)  # [N, 1, H, W]

    rows = []
    for i, name in enumerate(tqdm(test_dataset.image_names, desc="Encoding")):
        prob = all_probs[i : i + 1]  # [1, 1, 512, 512]
        orig_w, orig_h = original_sizes[i]

        resized = F.interpolate(
            prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        mask = (resized.squeeze().numpy() > args.threshold).astype(np.uint8)
        rows.append((name, rle_encode(mask)))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
