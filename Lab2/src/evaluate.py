import argparse
import torch
from tqdm import tqdm

from oxford_pet import get_dataloader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import dice_score, load_checkpoint, visualize_predictions, tta_predict


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
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
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--tta", action="store_true", help="Enable test-time augmentation"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save sample prediction visualizations"
    )
    parser.add_argument(
        "--vis_path",
        type=str,
        default="vis_eval.png",
        help="Path to save visualization",
    )
    return parser.parse_args()


def evaluate(model, dataloader, device, use_tta=False):
    model.eval()
    total_dice = 0.0
    total_images = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            if use_tta:
                probs = tta_predict(model, images)
            else:
                probs = torch.sigmoid(model(images))

            batch_dices = dice_score(probs, masks)
            total_dice += batch_dices.sum().item()
            total_images += images.size(0)

    return total_dice / total_images


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "unet":
        model = UNet(in_channels=3, num_classes=1).to(device)
    else:
        model = ResNet34UNet(in_channels=3, num_classes=1).to(device)

    load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint: {args.checkpoint}")

    _, val_loader = get_dataloader(
        args.data_root, args.batch_size, args.image_size, args.num_workers
    )

    avg_dice = evaluate(model, val_loader, device, use_tta=args.tta)
    print(f"Average Dice Score (val): {avg_dice:.4f}")
    if args.tta:
        print("(with TTA enabled)")

    if args.visualize:
        model.eval()
        with torch.no_grad():
            images, masks = next(iter(val_loader))
            images = images.to(device)
            masks = masks.to(device)
            if args.tta:
                probs = tta_predict(model, images)
            else:
                probs = torch.sigmoid(model(images))
            visualize_predictions(
                images, masks, probs, num_samples=4, save_path=args.vis_path
            )
            print(f"Visualization saved to {args.vis_path}")


if __name__ == "__main__":
    main()
