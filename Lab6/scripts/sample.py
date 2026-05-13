import argparse
from pathlib import Path

import torch
from diffusers.training_utils import EMAModel

from src.dataset import load_test_labels
from src.diffusion import build_ddim_scheduler
from src.evaluator_wrap import Evaluator
from src.model import ConditionalUNet
from src.sampler import sample_with_cfg, sample_with_cfg_and_classifier_guidance
from src.utils import save_image_grid, save_individual_images, set_seed


def load_unet_from_ckpt(ckpt_path: str, device: str = "cuda"):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]
    unet = ConditionalUNet(
        num_classes=cfg["num_classes"],
        sample_size=cfg["image_size"],
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg.get("layers_per_block", 2),
    ).to(device)
    ema = EMAModel(unet.parameters(), decay=cfg["ema_decay"])
    ema.load_state_dict(state["ema"])
    ema.copy_to(unet.parameters())
    unet.eval()
    return unet, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--num-inference-steps", type=int, default=100)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--classifier-guidance-scale", type=float, default=0.0)
    p.add_argument("--classifier-guidance-start", type=float, default=0.5)
    p.add_argument("--classifier-guidance-stop", type=float, default=1.0)
    p.add_argument("--classifier-bce", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", nargs="+", default=["test", "new_test"])
    args = p.parse_args()

    set_seed(args.seed)
    unet, cfg = load_unet_from_ckpt(args.ckpt)
    scheduler = build_ddim_scheduler()
    evaluator = Evaluator()

    out_root = Path(args.out_dir)
    for split in args.splits:
        labels = load_test_labels(args.data_root, split, cfg["num_classes"])
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        if args.classifier_guidance_scale == 0:
            images = sample_with_cfg(
                unet,
                scheduler,
                labels,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
                image_size=cfg["image_size"],
            )
        else:
            images = sample_with_cfg_and_classifier_guidance(
                unet,
                scheduler,
                labels,
                classifier=evaluator.resnet18,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                classifier_guidance_scale=args.classifier_guidance_scale,
                classifier_guidance_start=args.classifier_guidance_start,
                classifier_guidance_stop=args.classifier_guidance_stop,
                classifier_positive_only=not args.classifier_bce,
                generator=gen,
                image_size=cfg["image_size"],
            )
        acc = evaluator.eval(images, labels)
        print(f"[{split}] acc = {acc:.4f}")

        save_image_grid(
            images.cpu(), out_root / f"{split}_grid.png", nrow=8
        )
        save_individual_images(
            images.cpu(), out_root / "images" / split, name_fmt="{idx:03d}.png"
        )


if __name__ == "__main__":
    main()
