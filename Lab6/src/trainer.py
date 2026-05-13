import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .dataset import ICLEVRDataset, load_test_labels
from .diffusion import build_ddim_scheduler, build_train_scheduler
from .evaluator_wrap import Evaluator
from .sampler import sample_with_cfg, sample_with_cfg_and_classifier_guidance
from .utils import save_image_grid, set_seed


@dataclass
class TrainConfig:
    data_root: str = "data"
    out_dir: str = "checkpoints"
    image_size: int = 64
    num_classes: int = 24
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512)
    layers_per_block: int = 2
    batch_size: int = 128
    num_workers: int = 8
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    cfg_dropout_prob: float = 0.1
    ema_decay: float = 0.9999
    eval_every: int = 5
    eval_inference_steps: int = 50
    eval_guidance_scale: float = 3.0
    eval_classifier_guidance_scale: float = 0.0
    eval_classifier_guidance_start: float = 0.0
    eval_classifier_guidance_stop: float = 0.9
    eval_classifier_positive_only: bool = False
    grad_clip: float = 1.0
    seed: int = 42
    mixed_precision: bool = True
    lr_scheduler: str = "constant"
    onecycle_pct_start: float = 0.05
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1000.0
    wandb_project: str = "dlp-lab6"
    wandb_run_name: str | None = None


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device("cuda")
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = ICLEVRDataset(
            cfg.data_root,
            image_size=cfg.image_size,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        )

        from .model import ConditionalUNet

        self.unet = ConditionalUNet(
            num_classes=cfg.num_classes,
            sample_size=cfg.image_size,
            block_out_channels=cfg.block_out_channels,
            layers_per_block=cfg.layers_per_block,
        ).to(self.device)
        self.ema = EMAModel(self.unet.parameters(), decay=cfg.ema_decay)
        self.ema.to(self.device)

        self.train_scheduler: DDPMScheduler = build_train_scheduler()
        self.eval_scheduler = build_ddim_scheduler()

        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.lr_scheduler = None
        if cfg.lr_scheduler == "onecycle":
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=cfg.lr,
                total_steps=cfg.epochs * len(self.loader),
                pct_start=cfg.onecycle_pct_start,
                div_factor=cfg.onecycle_div_factor,
                final_div_factor=cfg.onecycle_final_div_factor,
                anneal_strategy="cos",
            )
        elif cfg.lr_scheduler != "constant":
            raise ValueError(f"unsupported lr_scheduler: {cfg.lr_scheduler}")
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)

        self.evaluator = Evaluator()
        self.test_labels = load_test_labels(cfg.data_root, "test", cfg.num_classes)
        self.new_test_labels = load_test_labels(
            cfg.data_root, "new_test", cfg.num_classes
        )

        self.global_step = 0
        self.best_acc = -1.0

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg.__dict__,
        )

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # CFG dropout: replace condition with zeros
        drop_mask = (
            torch.rand(y.size(0), device=self.device) < self.cfg.cfg_dropout_prob
        )
        if drop_mask.any():
            y = y.clone()
            y[drop_mask] = 0.0

        t = torch.randint(
            0,
            self.train_scheduler.config.num_train_timesteps,
            (x.size(0),),
            device=self.device,
        ).long()
        noise = torch.randn_like(x)
        x_t = self.train_scheduler.add_noise(x, noise, t)

        with torch.amp.autocast(
            device_type="cuda", enabled=self.cfg.mixed_precision, dtype=torch.float16
        ):
            eps_pred = self.unet(x_t, t, y)
            loss = F.mse_loss(eps_pred, noise)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.unet.parameters(), self.cfg.grad_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.ema.step(self.unet.parameters())

        return float(loss.detach().item())

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict[str, float]:
        self.ema.store(self.unet.parameters())
        self.ema.copy_to(self.unet.parameters())
        self.unet.eval()

        results: dict[str, float] = {}
        for split, labels in [
            ("test", self.test_labels),
            ("new_test", self.new_test_labels),
        ]:
            # Fixed seed for eval to reduce variance when comparing checkpoints
            gen = torch.Generator(device=self.device).manual_seed(1234)
            if self.cfg.eval_classifier_guidance_scale == 0:
                images = sample_with_cfg(
                    self.unet,
                    self.eval_scheduler,
                    labels,
                    num_inference_steps=self.cfg.eval_inference_steps,
                    guidance_scale=self.cfg.eval_guidance_scale,
                    device=self.device,
                    generator=gen,
                    image_size=self.cfg.image_size,
                )
            else:
                images = sample_with_cfg_and_classifier_guidance(
                    self.unet,
                    self.eval_scheduler,
                    labels,
                    classifier=self.evaluator.resnet18,
                    num_inference_steps=self.cfg.eval_inference_steps,
                    guidance_scale=self.cfg.eval_guidance_scale,
                    classifier_guidance_scale=self.cfg.eval_classifier_guidance_scale,
                    classifier_guidance_start=self.cfg.eval_classifier_guidance_start,
                    classifier_guidance_stop=self.cfg.eval_classifier_guidance_stop,
                    classifier_positive_only=self.cfg.eval_classifier_positive_only,
                    generator=gen,
                    device=str(self.device),
                    image_size=self.cfg.image_size,
                )
            acc = self.evaluator.eval(images, labels)
            results[f"{split}_acc"] = acc
            grid_path = self.out_dir / f"grids/epoch_{epoch:04d}_{split}.png"
            save_image_grid(images.cpu(), grid_path, nrow=8)

        self.ema.restore(self.unet.parameters())
        self.unet.train()
        return results

    def save_checkpoint(self, name: str, extra: dict | None = None) -> None:
        state = {
            "unet": self.unet.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.cfg.__dict__,
            "global_step": self.global_step,
        }
        if extra:
            state.update(extra)
        torch.save(state, self.out_dir / f"{name}.pt")

    def fit(self) -> None:
        self.unet.train()
        for epoch in range(1, self.cfg.epochs + 1):
            pbar = tqdm(self.loader, desc=f"epoch {epoch}/{self.cfg.epochs}")
            losses: list[float] = []
            for x, y in pbar:
                loss = self._train_step(x, y)
                losses.append(loss)
                self.global_step += 1
                if self.global_step % 50 == 0:
                    wandb.log(
                        {"train/loss": loss, "epoch": epoch}, step=self.global_step
                    )
                pbar.set_postfix(loss=f"{loss:.4f}")

            avg = sum(losses) / max(1, len(losses))
            wandb.log(
                {"train/loss_epoch": avg, "epoch": epoch}, step=self.global_step
            )

            if epoch % self.cfg.eval_every == 0 or epoch == self.cfg.epochs:
                metrics = self.evaluate(epoch)
                wandb.log(
                    {f"eval/{k}": v for k, v in metrics.items()} | {"epoch": epoch},
                    step=self.global_step,
                )
                avg_acc = sum(metrics.values()) / len(metrics)
                if avg_acc > self.best_acc:
                    self.best_acc = avg_acc
                    self.save_checkpoint("best", extra={"metrics": metrics})
                self.save_checkpoint("last", extra={"metrics": metrics})
                tqdm.write(
                    f"[epoch {epoch}] "
                    + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    + f" | best_avg={self.best_acc:.4f}"
                )
