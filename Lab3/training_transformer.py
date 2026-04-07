import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
import wandb
from torch.utils.data import DataLoader


# TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS, train_loader=None, val_loader=None):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(
            device=args.device
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim, self.scheduler = self.configure_optimizers(args)
        self.prepare_training(args.save_dir)

    @staticmethod
    def prepare_training(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    def train_one_epoch(self, args):
        self.model.train()
        total_loss = 0
        self.optim.zero_grad()
        for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            images = batch.to(device=args.device)
            logits, z_indices = self.model(images)
            logits = logits.permute(
                0, 2, 1
            )  # (B, num_codebook_vectors, N=256) for cross_entropy
            loss = F.cross_entropy(logits, z_indices) / args.accum_grad
            loss.backward()

            # Gradient accumulation
            if (step + 1) % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

            total_loss += loss.item() * args.accum_grad

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def eval_one_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                images = batch.to(device=self.optim.param_groups[0]["params"][0].device)
                logits, z_indices = self.model(images)
                logits = logits.permute(
                    0, 2, 1
                )  # (B, num_codebook_vectors, N=256) for cross_entropy
                loss = F.cross_entropy(
                    logits, z_indices
                )  # (B, N=256, num_codebook_vectors)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def configure_optimizers(self, args):
        optimizer = torch.optim.AdamW(
            self.model.transformer.parameters(), lr=args.learning_rate
        )
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )
        elif args.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
        elif args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.learning_rate,
                steps_per_epoch=len(self.train_loader),
                epochs=args.epochs,
                pct_start=0.1,
                anneal_strategy="cos",
            )
        return optimizer, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskGIT")
    # TODO2:check your dataset path is correct
    parser.add_argument(
        "--train_d_path",
        type=str,
        default="./lab3_dataset/train/",
        help="Training Dataset Path",
    )
    parser.add_argument(
        "--val_d_path",
        type=str,
        default="./lab3_dataset/val/",
        help="Validation Dataset Path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoints/last_ckpt.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Which device the training is on."
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.0-1.0), useful for debugging.",
    )
    parser.add_argument(
        "--accum-grad", type=int, default=2, help="Number for gradient accumulation."
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train."
    )
    parser.add_argument(
        "--save-per-epoch", type=int, default=20, help="Save CKPT per N epochs."
    )
    parser.add_argument(
        "--start-from-epoch",
        type=int,
        default=0,
        help="Resume training from this epoch.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate."
    )

    parser.add_argument(
        "--MaskGitConfig",
        type=str,
        default="config/MaskGit.yml",
        help="Configurations for TransformerVQGAN",
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine", help="Scheduler for training"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./models/Transformer/checkpoints/", help="Directory to save checkpoints."
    )

    args = parser.parse_args()

    wandb.init(
        project="nycu-dlp-lab3",
        name=f"transformer_{args.lr_scheduler}_{args.epochs}epochs_{args.batch_size}bs",
        config=vars(args),
    )

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, "r"))
    train_transformer = TrainTransformer(
        args, MaskGit_CONFIGS, train_loader, val_loader
    )

    # TODO2 step1-5:
    best_val_loss = float("inf")
    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
        train_loss = train_transformer.train_one_epoch(args)
        val_loss = train_transformer.eval_one_epoch()
        if train_transformer.scheduler is not None:
            if args.lr_scheduler == "plateau":
                train_transformer.scheduler.step(val_loss)
            elif args.lr_scheduler == "cosine":
                train_transformer.scheduler.step()

        print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}")

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": train_transformer.optim.param_groups[0]["lr"],
            }
        )

        if epoch % args.save_per_epoch == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pt")
            torch.save(train_transformer.model.transformer.state_dict(), ckpt_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.save_dir, "best_ckpt.pt")
            torch.save(train_transformer.model.transformer.state_dict(), ckpt_path)

    wandb.finish()
