import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  # Senior jayin92's recipe: clamp logvar + normalize by total elements
  # (B * N_dim * H * W). This ~25k× scale fix is the single biggest gap
  # vs the broken template — without it, β=1 KL crushes the MSE term.
  # Tighter clamp [-5, 5]: prevents logvar.exp() blowing up to ~22000
  # which can cause NaN in the autoregressive rollout.
  logvar = torch.clamp(logvar, min=-5.0, max=5.0)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  total_elements = batch_size * mu.shape[1] * mu.shape[2] * mu.shape[3]
  KLD = KLD / max(1, total_elements)
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.type = args.kl_anneal_type
        self.n_epoch = args.num_epoch
        self.n_cycle = max(1, args.kl_anneal_cycle)
        self.ratio = args.kl_anneal_ratio
        self.i = current_epoch

        if self.type == 'Cyclical':
            self.L = self.frange_cycle_linear(
                self.n_epoch, start=0.0, stop=1.0,
                n_cycle=self.n_cycle, ratio=self.ratio,
            )
        elif self.type == 'Monotonic':
            # Single 0->1 ramp over `ratio` of training, then hold at 1.
            self.L = self.frange_cycle_linear(
                self.n_epoch, start=0.0, stop=1.0,
                n_cycle=1, ratio=self.ratio,
            )
        else:  # 'None' or unknown -> constant beta=1
            self.L = np.ones(self.n_epoch, dtype=np.float32)

    def update(self):
        self.i += 1

    def get_beta(self):
        idx = min(self.i, len(self.L) - 1)
        return float(self.L[idx])

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1.0):
        """
        Cyclical linear annealing schedule (Fu et al. 2019).
        For each of `n_cycle` cycles of length ~n_iter/n_cycle, beta ramps
        linearly from `start` to `stop` over the first `ratio` fraction of the
        cycle, then stays at `stop` for the rest of the cycle.
        """
        L = np.ones(n_iter, dtype=np.float32) * stop
        period = n_iter / n_cycle
        step = (stop - start) / max(1.0, period * ratio)  # per-epoch increment

        for c in range(n_cycle):
            v, j = start, 0
            while v <= stop and int(c * period + j) < n_iter:
                L[int(c * period + j)] = v
                v += step
                j += 1
        return L

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = self._build_scheduler(self.optim, self.args)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        # Init wandb run once, at the start of training.
        if _WANDB_AVAILABLE and getattr(self.args, 'use_wandb', True):
            if wandb.run is None:
                wandb.init(
                    project="dlp-lab4",
                    name=getattr(self.args, 'run_name', None),
                    config=vars(self.args),
                )
                wandb.watch(self, log=None)
            # Use `epoch` as x-axis for val/* and train/epoch_* metrics; keep
            # default step axis for per-batch train/step_* metrics.
            wandb.define_metric("epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("train/epoch_*", step_metric="epoch")
            wandb.define_metric("train/beta", step_metric="epoch")
            wandb.define_metric("train/tfr", step_metric="epoch")
            wandb.define_metric("train/lr", step_metric="epoch")

        global_step = 0
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()

            epoch_losses, epoch_mses, epoch_klds = [], [], []

            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                # Per-batch TF coin flip (was per-epoch). Per-epoch makes the
                # entire epoch all-TF or all-no-TF — extremely noisy gradients
                # and the source of the abrupt PSNR cliff at epoch ~19.
                adapt_TeacherForcing = random.random() < self.tfr
                loss, mse_val, kld_val = self.training_one_step(
                    img, label, adapt_TeacherForcing
                )

                beta = self.kl_annealing.get_beta()
                epoch_losses.append(float(loss.detach().cpu()))
                epoch_mses.append(mse_val)
                epoch_klds.append(kld_val)

                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.3f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.3f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

                if _WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        "train/step_loss": float(loss.detach().cpu()),
                        "train/step_mse":  mse_val,
                        "train/step_kld":  kld_val,
                        "train/beta":      beta,
                        "train/tfr":       self.tfr,
                        "train/tf_on":     int(adapt_TeacherForcing),
                        "train/lr":        self.scheduler.get_last_lr()[0],
                        "epoch":           self.current_epoch,
                    }, step=global_step)
                global_step += 1

            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))

            val_loss, val_psnr = self.eval()

            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "train/epoch_loss":  float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                    "train/epoch_mse":   float(np.mean(epoch_mses))   if epoch_mses   else 0.0,
                    "train/epoch_kld":   float(np.mean(epoch_klds))   if epoch_klds   else 0.0,
                    "train/beta":        self.kl_annealing.get_beta(),
                    "train/tfr":         self.tfr,
                    "train/lr":          self.scheduler.get_last_lr()[0],
                    "val/loss":          val_loss,
                    "val/psnr":          val_psnr,
                    "epoch":             self.current_epoch,
                })

            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()


    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        losses, psnrs = [], []
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)
            losses.append(float(loss.detach().cpu()))
            psnrs.append(float(psnr))
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_psnr = float(np.mean(psnrs)) if psnrs else 0.0
        print(f"[val] epoch={self.current_epoch}  loss={mean_loss:.4f}  PSNR={mean_psnr:.2f} dB")
        return mean_loss, mean_psnr

    def training_one_step(self, img, label, adapt_TeacherForcing):
        """
        img, label : (B, T, C, H, W)
        Returns: (total_loss_tensor, mean_mse_float, mean_kld_float)
        """
        # Move time to the front: (T, B, C, H, W)
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        T, B = img.shape[0], img.shape[1]

        self.optim.zero_grad()

        beta = self.kl_annealing.get_beta()
        total_mse = 0.0
        total_kld = 0.0

        prev_frame = img[0]            # start from the first ground-truth frame
        last_pred = img[0]             # used when teacher-forcing is off

        for t in range(1, T):
            cur_gt = img[t]
            cur_pose = label[t]

            if adapt_TeacherForcing:
                prev_input = img[t - 1]
            else:
                prev_input = last_pred

            enc_prev = self.frame_transformation(prev_input)
            enc_curr = self.frame_transformation(cur_gt)
            enc_pose = self.label_transformation(cur_pose)

            z, mu, logvar = self.Gaussian_Predictor(enc_curr, enc_pose)
            fused = self.Decoder_Fusion(enc_prev, enc_pose, z)
            pred = self.Generator(fused)
            # NO NaN→0.5 mask in training: that path has zero gradient and
            # creates an absorbing constant-output attractor (PSNR=15.28 trap).
            # Tighter logvar clamp + grad-clip is enough; if NaNs still occur
            # we want loss.backward() to fail loudly rather than silently
            # train the model toward a gray frame.

            mse = self.mse_criterion(pred.clamp(0.0, 1.0), cur_gt)
            kld = kl_criterion(mu, logvar, B)

            total_mse = total_mse + mse
            total_kld = total_kld + kld

            # Always detach for the next-step input — keeping grads makes
            # BPTT span the full T-1 rollout, which explodes once TF turns
            # OFF. Senior jayin92's recipe.
            last_pred = pred.detach().clamp(0.0, 1.0)

        # Senior jayin92's recipe: SUM over time (not mean). The relative
        # MSE/KL balance and absolute gradient magnitude both matter.
        if beta > 0:
            loss = total_mse + beta * total_kld
        else:
            loss = total_mse

        loss.backward()
        self.optimizer_step()

        steps = max(1, T - 1)
        return loss, float((total_mse / steps).detach().cpu()), float((total_kld / steps).detach().cpu())

    def val_one_step(self, img, label):
        """
        Validation: autoregressive generation, no teacher forcing, z ~ N(0, I).
        Returns (loss_tensor, psnr_float).
        """
        img = img.permute(1, 0, 2, 3, 4)     # (T, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4)
        T, B, C, H, W = img.shape

        total_mse = 0.0
        psnrs = []
        prev = img[0]

        for t in range(1, T):
            cur_gt = img[t]
            cur_pose = label[t]

            enc_prev = self.frame_transformation(prev)
            enc_pose = self.label_transformation(cur_pose)

            # Sample z from the prior at inference time
            # Gaussian_Predictor output has N_dim channels at the same H, W as enc_prev
            # z=0 (prior mean) at val matches our Tester.py inference setting.
            # randn injects per-frame noise that destroys autoregressive PSNR.
            z = torch.zeros(B, self.args.N_dim, enc_prev.shape[-2], enc_prev.shape[-1],
                            device=img.device, dtype=enc_prev.dtype)

            fused = self.Decoder_Fusion(enc_prev, enc_pose, z)
            pred = self.Generator(fused)
            pred = pred.clamp(0.0, 1.0)
            pred = torch.nan_to_num(pred, nan=0.5)

            mse = self.mse_criterion(pred, cur_gt)
            total_mse = total_mse + mse
            psnrs.append(float(Generate_PSNR(pred, cur_gt).detach().cpu()))

            prev = pred

        steps = max(1, T - 1)
        loss = total_mse / steps
        mean_psnr = float(np.mean(psnrs)) if psnrs else 0.0
        return loss, mean_psnr
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        """
        After the start-decay epoch (args.tfr_sde), decrease tfr by tfr_d_step
        each epoch, clamped to [0, 1].
        """
        if self.current_epoch >= self.args.tfr_sde:
            self.tfr = max(0.0, self.tfr - self.args.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = self._build_scheduler(self.optim, self.args)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

    @staticmethod
    def _build_scheduler(optimizer, args):
        """
        LR scheduler factory. Selected via --lr_scheduler.

        multistep   : MultiStepLR, milestones=[2,5], gamma=0.1  (template default)
        cosine      : CosineAnnealingLR over num_epoch
        cosine_warm : CosineAnnealingWarmRestarts (T_0 = num_epoch//4)
        step        : StepLR(step_size=10, gamma=0.5)
        none        : LambdaLR identity (constant lr)
        """
        kind = getattr(args, 'lr_scheduler', 'multistep')
        if kind == 'multistep':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5], gamma=0.1)
        if kind == 'cosine':
            # eta_min floor matches senior's recipe (default 0.0)
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.num_epoch), eta_min=args.lr_min)
        if kind == 'cosine_warm':
            T_0 = max(1, args.num_epoch // 4)
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2)
        if kind == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        if kind == 'none':
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _e: 1.0)
        raise ValueError(f"Unknown --lr_scheduler: {kind}")



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=12)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--lr_min',        type=float,  default=0.0,       help="cosine scheduler eta_min floor")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,            help="")

    # LR scheduler choice
    parser.add_argument('--lr_scheduler',       type=str, default='cosine',
                        choices=['multistep', 'cosine', 'cosine_warm', 'step', 'none'],
                        help="Learning-rate scheduler kind.")

    # wandb logging
    parser.add_argument('--use_wandb',          action='store_true', default=True,
                        help="Log training metrics to Weights & Biases (project 'dlp-lab4').")
    parser.add_argument('--no_wandb',           dest='use_wandb', action='store_false',
                        help="Disable wandb logging.")
    parser.add_argument('--run_name',           type=str, default=None,
                        help="Optional wandb run name.")
    

    

    args = parser.parse_args()
    
    main(args)
