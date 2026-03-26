import torch
import numpy as np
import matplotlib.pyplot as plt


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute per-image Dice scores for binary segmentation.

    Returns:
        1D tensor of shape [B] — one dice score per image (NOT averaged).
        Caller is responsible for accumulating and averaging across the full dataset.
    """
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
    )
    return dice


def dice_loss(pred, target, smooth=1e-6):
    """
    Differentiable Dice Loss = 1 - soft Dice score.

    Args:
        pred:   soft probabilities after sigmoid [B, 1, H, W] (NOT binarized)
        target: ground truth binary mask [B, 1, H, W]
        smooth: small constant to avoid division by zero

    Returns:
        scalar tensor: mean Dice loss
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()


def save_checkpoint(model, optimizer, epoch, path, **kwargs):
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def visualize_predictions(images, masks, preds, num_samples=4, save_path=None):
    """
    Args:
        images: [B, 3, H, W] normalized tensor
        masks:  [B, 1, H, W] binary ground truth
        preds:  [B, 1, H, W] predicted probabilities
        num_samples: how many to plot
        save_path: optional path to save figure
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    n = min(num_samples, images.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.unsqueeze(0) if hasattr(axes, "unsqueeze") else [axes]

    for i in range(n):
        img = images[i].cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        gt = masks[i, 0].cpu().numpy()
        pred = (preds[i, 0].cpu() > 0.5).float().numpy()

        axes[i][0].imshow(img)
        axes[i][0].set_title("Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")

        axes[i][2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[i][2].set_title("Prediction")
        axes[i][2].axis("off")

    plt.tight_layout()
    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def rle_encode(mask):
    """
    Encode a binary mask to Kaggle RLE format.

    Convention: 1-indexed, column-major (Fortran) order, space-delimited pairs.

    Args:
        mask: 2D numpy array (H x W) with values 0 or 1

    Returns:
        RLE string "start1 length1 start2 length2 ..."
        Empty string for all-zero masks.
    """
    pixels = mask.T.flatten()  # column-major (Fortran) order
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # 1-indexed
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)


def tta_predict(model, images):
    """
    4× TTA: average logits over original, h-flip, v-flip, and h+v-flip.

    Returns:
        [B, 1, H, W] sigmoid probabilities
    """
    logits_orig = model(images)

    flipped_h = torch.flip(images, dims=[-1])
    logits_h = torch.flip(model(flipped_h), dims=[-1])

    flipped_v = torch.flip(images, dims=[-2])
    logits_v = torch.flip(model(flipped_v), dims=[-2])

    flipped_hv = torch.flip(images, dims=[-1, -2])
    logits_hv = torch.flip(model(flipped_hv), dims=[-1, -2])

    avg_logits = (logits_orig + logits_h + logits_v + logits_hv) / 4.0
    return torch.sigmoid(avg_logits)
