import argparse

from src.trainer import TrainConfig, Trainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--cfg-dropout-prob", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-inference-steps", type=int, default=100)
    p.add_argument("--eval-guidance-scale", type=float, default=3.0)
    p.add_argument("--eval-classifier-guidance-scale", type=float, default=0.0)
    p.add_argument("--eval-classifier-guidance-start", type=float, default=0.5)
    p.add_argument("--eval-classifier-guidance-stop", type=float, default=1.0)
    p.add_argument("--eval-classifier-bce", action="store_true", default=True)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--lr-scheduler", choices=["constant", "onecycle"], default="onecycle")
    p.add_argument("--onecycle-pct-start", type=float, default=0.05)
    p.add_argument("--onecycle-div-factor", type=float, default=25.0)
    p.add_argument("--onecycle-final-div-factor", type=float, default=1000.0)
    p.add_argument("--wandb-project", default="dlp-lab6")
    p.add_argument("--run-name", default=None, help="wandb run name")
    p.add_argument(
        "--block-out-channels",
        type=int,
        nargs="+",
        default=[128, 256, 256, 512],
    )
    p.add_argument("--layers-per-block", type=int, default=2)
    args = p.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        image_size=args.image_size,
        block_out_channels=tuple(args.block_out_channels),
        layers_per_block=args.layers_per_block,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cfg_dropout_prob=args.cfg_dropout_prob,
        ema_decay=args.ema_decay,
        eval_every=args.eval_every,
        eval_inference_steps=args.eval_inference_steps,
        eval_guidance_scale=args.eval_guidance_scale,
        eval_classifier_guidance_scale=args.eval_classifier_guidance_scale,
        eval_classifier_guidance_start=args.eval_classifier_guidance_start,
        eval_classifier_guidance_stop=args.eval_classifier_guidance_stop,
        eval_classifier_positive_only=not args.eval_classifier_bce,
        grad_clip=args.grad_clip,
        seed=args.seed,
        mixed_precision=not args.no_amp,
        lr_scheduler=args.lr_scheduler,
        onecycle_pct_start=args.onecycle_pct_start,
        onecycle_div_factor=args.onecycle_div_factor,
        onecycle_final_div_factor=args.onecycle_final_div_factor,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
    )
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
