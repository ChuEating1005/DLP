"""Ensemble inference: average predictions across N stochastic-z rollouts.

For each test sequence, we run N independent autoregressive rollouts (each
sampling its own z ~ N(0, I) * z_scale per frame), then average the float
predictions per pixel before quantizing to uint8 for the submission CSV.

Averaging in float space typically gains +0.5-1 dB by suppressing noise
introduced by random z while preserving the mean prediction.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from torch import stack
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder
from Trainer import VAE_Model
from Tester import Dataset_Dance, Test_model
import pandas as pd


class EnsembleTester(Test_model):
    @torch.no_grad()
    def rollout(self, img, label, z_scale):
        # img: (1, 1, 3, H, W) -> (1, B=1, 3, H, W) seed only; label: (1, 630, ...)
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        assert label.shape[0] == 630
        prev = img[0]
        out = []
        for t in range(label.shape[0]):
            cur_pose = label[t]
            enc_prev = self.frame_transformation(prev)
            enc_pose = self.label_transformation(cur_pose)
            z = torch.randn(
                prev.shape[0], self.args.N_dim,
                enc_prev.shape[-2], enc_prev.shape[-1],
                device=prev.device, dtype=enc_prev.dtype,
            ) * z_scale
            fused = self.Decoder_Fusion(enc_prev, enc_pose, z)
            pred = self.Generator(fused)
            pred = torch.clamp(pred, 0.0, 1.0)
            out.append(pred.cpu())
            prev = pred
        return stack(out)  # (630, 1, 3, H, W)

    @torch.no_grad()
    def eval_ensemble(self, n_runs, z_scale, seed):
        loader = self.val_dataloader()
        all_seqs = []  # list per sequence: (630, 1, 3, H, W) accumulator
        for idx, (img, label) in enumerate(tqdm(loader, ncols=80, desc='seqs')):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            acc = None
            for r in range(n_runs):
                torch.manual_seed(seed + 1000 * idx + r)
                pred = self.rollout(img, label, z_scale)  # float in [0,1]
                acc = pred if acc is None else acc + pred
            mean_pred = acc / n_runs  # (630, 1, 3, H, W)
            # reshape to (630, 3*64*32) like baseline
            mean_pred = mean_pred.permute(1, 0, 2, 3, 4)  # (1, 630, 3, H, W)
            assert mean_pred.shape == (1, 630, 3, 32, 64)
            all_seqs.append(mean_pred[0].reshape(630, -1))

        full = torch.cat(all_seqs)  # (5*630, 3*64*32) float
        pred_to_int = (np.rint(full.numpy() * 255)).astype(int)
        df = pd.DataFrame(pred_to_int)
        df.insert(0, 'id', range(0, len(df)))
        out_csv = os.path.join(self.args.save_root, 'submission.csv')
        df.to_csv(out_csv, header=True, index=False)
        print(f'[ensemble] wrote {out_csv}  rows={len(df)}  n_runs={n_runs}  z_scale={z_scale}')


def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = EnsembleTester(args).to(args.device)
    model.load_checkpoint()
    model.eval_ensemble(n_runs=args.n_runs, z_scale=args.z_scale, seed=args.seed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--optim', type=str, default='Adam')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--test', action='store_true')
    p.add_argument('--no_sanity', action='store_true')
    p.add_argument('--make_gif', action='store_true')
    p.add_argument('--DR', type=str, required=True)
    p.add_argument('--save_root', type=str, required=True)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--num_epoch', type=int, default=70)
    p.add_argument('--per_save', type=int, default=3)
    p.add_argument('--partial', type=float, default=1.0)
    p.add_argument('--train_vi_len', type=int, default=16)
    p.add_argument('--val_vi_len', type=int, default=630)
    p.add_argument('--frame_H', type=int, default=32)
    p.add_argument('--frame_W', type=int, default=64)
    p.add_argument('--F_dim', type=int, default=128)
    p.add_argument('--L_dim', type=int, default=32)
    p.add_argument('--N_dim', type=int, default=12)
    p.add_argument('--D_out_dim', type=int, default=192)
    p.add_argument('--tfr', type=float, default=1.0)
    p.add_argument('--tfr_sde', type=int, default=10)
    p.add_argument('--tfr_d_step', type=float, default=0.1)
    p.add_argument('--ckpt_path', type=str, default=None)
    p.add_argument('--fast_train', action='store_true')
    p.add_argument('--fast_partial', type=float, default=0.4)
    p.add_argument('--fast_train_epoch', type=int, default=5)
    p.add_argument('--kl_anneal_type', type=str, default='Cyclical')
    p.add_argument('--kl_anneal_cycle', type=int, default=10)
    p.add_argument('--kl_anneal_ratio', type=float, default=1)
    # ensemble args
    p.add_argument('--n_runs', type=int, default=10)
    p.add_argument('--z_scale', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main(args)
