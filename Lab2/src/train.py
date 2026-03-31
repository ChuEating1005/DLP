import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from oxford_pet import get_dataloader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import (
    dice_score,
    dice_loss,
    save_checkpoint,
    tta_predict,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "resnet34_unet"],
        help="Model architecture",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to dataset root (default: auto-detect)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Resize dimension (default: 260 for unet, 256 for resnet34_unet)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models/",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation for validation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        choices=["onecycle", "cosine", "plateau"],
        help="LR scheduler: onecycle (per-batch), cosine (per-epoch), or plateau (per-epoch)",
    )
    return parser.parse_args()


def get_model(model_name, device):
    if model_name == "unet":
        model = UNet(in_channels=3, num_classes=1)
    elif model_name == "resnet34_unet":
        model = ResNet34UNet(in_channels=3, num_classes=1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    return model.to(device)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    total_dice = 0.0
    total_images = 0
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        batch_dices = dice_score(torch.sigmoid(logits), masks)
        total_dice += batch_dices.sum().item()
        total_images += images.size(0)
        pbar.set_postfix(loss=loss.item(), dice=batch_dices.mean().item())

    avg_loss = running_loss / len(dataloader)
    avg_dice = total_dice / total_images
    return avg_loss, avg_dice


def validate(model, dataloader, criterion, device, use_tta=False):
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_images = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Val", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item()

            if use_tta:
                probs = tta_predict(model, images)
            else:
                probs = torch.sigmoid(logits)
            batch_dices = dice_score(probs, masks)
            total_dice += batch_dices.sum().item()
            total_images += images.size(0)

    avg_loss = running_loss / len(dataloader)
    avg_dice = total_dice / total_images
    return avg_loss, avg_dice


def tune_threshold(model, dataloader, device, use_tta=False):
    model.eval()
    all_probs = []
    all_masks = []
    with torch.no_grad():
        for images, masks in tqdm(
            dataloader, desc="Collecting for threshold tuning", leave=False
        ):
            images = images.to(device)
            masks = masks.to(device)
            if use_tta:
                probs = tta_predict(model, images)
            else:
                probs = torch.sigmoid(model(images))
            all_probs.append(probs)
            all_masks.append(masks)

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    best_thresh = 0.5
    best_dice = 0.0
    for t in [i * 0.05 for i in range(6, 15)]:
        d = dice_score(all_probs, all_masks, threshold=t).mean().item()
        tqdm.write(f"  threshold={t:.2f} -> dice={d:.4f}")
        if d > best_dice:
            best_dice = d
            best_thresh = t

    return best_thresh, best_dice


def main():
    args = parse_args()
    if args.image_size is None:
        args.image_size = 260 if args.model == "unet" else 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    if args.wandb:
        import wandb

        wandb.init(
            project="nycu-dlp-lab2",
            name=f"{args.model}_lr{args.lr}_e{args.epochs}_res{args.image_size}_bs{args.batch_size}_{args.scheduler}",
            config=vars(args),
        )

    train_loader, val_loader = get_dataloader(
        args.data_root, args.batch_size, args.image_size, args.num_workers
    )
    model = get_model(args.model, device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    batch_scheduler = None
    epoch_scheduler = None

    if args.scheduler == "onecycle":
        batch_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
        )
    elif args.scheduler == "cosine":
        epoch_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.scheduler == "plateau":
        epoch_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
        )

    criterion = lambda logits, masks: nn.BCEWithLogitsLoss()(logits, masks) + dice_loss(
        torch.sigmoid(logits), masks
    )

    best_dice = 0.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler=batch_scheduler,
        )
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, use_tta=args.tta
        )

        if epoch_scheduler is not None:
            if args.scheduler == "plateau":
                epoch_scheduler.step(val_dice)
            else:
                epoch_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        tqdm.write(
            f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}, train_dice={train_dice:.4f}, "
            f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}, lr={current_lr:.6f}"
        )

        if args.wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/dice": train_dice,
                    "val/loss": val_loss,
                    "val/dice": val_dice,
                    "lr": current_lr,
                }
            )

        if val_dice > best_dice:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                os.path.join(args.save_dir, f"{args.model}_best.pth"),
                best_dice=val_dice,
            )
            best_dice = val_dice
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            tqdm.write(
                f"Early stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)"
            )
            break

    tqdm.write(f"\nTraining complete. Best val dice: {best_dice:.4f}")

    tqdm.write("\n--- Threshold Tuning on Validation Set ---")
    checkpoint = torch.load(
        os.path.join(args.save_dir, f"{args.model}_best.pth"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    best_thresh, best_thresh_dice = tune_threshold(
        model, val_loader, device, use_tta=args.tta
    )
    tqdm.write(f"Best threshold: {best_thresh:.2f} (dice={best_thresh_dice:.4f})")

    if args.wandb:
        wandb.log({"best_val_dice": best_dice, "best_threshold": best_thresh})
        wandb.finish()


if __name__ == "__main__":
    main()
