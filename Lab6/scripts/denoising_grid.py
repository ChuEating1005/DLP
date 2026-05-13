import argparse
from pathlib import Path

import torch

from scripts.sample import load_unet_from_ckpt
from src.dataset import encode_labels
from src.diffusion import build_ddpm_scheduler
from src.sampler import sample_with_trace
from src.utils import save_image_grid, set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="outputs/denoising_process.png")
    p.add_argument("--num-inference-steps", type=int, default=1000)
    p.add_argument("--num-snapshots", type=int, default=8)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--labels",
        nargs="+",
        default=["red sphere", "cyan cylinder", "cyan cube"],
    )
    args = p.parse_args()

    set_seed(args.seed)
    unet, cfg = load_unet_from_ckpt(args.ckpt)
    scheduler = build_ddpm_scheduler()

    import json

    with open(Path(args.data_root) / "objects.json") as f:
        obj2idx = json.load(f)
    labels = encode_labels([args.labels], obj2idx, cfg["num_classes"])

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    _, trace = sample_with_trace(
        unet,
        scheduler,
        labels,
        num_inference_steps=args.num_inference_steps,
        num_snapshots=args.num_snapshots,
        guidance_scale=args.guidance_scale,
        generator=gen,
        image_size=cfg["image_size"],
    )

    frames = trace[:, 0]
    save_image_grid(frames.cpu(), args.out, nrow=args.num_snapshots)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
