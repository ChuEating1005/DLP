"""
Per-frame PSNR vs frame index on validation set.

Loads a trained VAE_Model checkpoint, autoregressively rolls out the val
sequence with z ~ N(0, I) * z_scale (default 0.5, matching our Tester),
records per-frame PSNR, and saves both a CSV and a PNG plot.

Usage:
  python src/plot_per_frame_psnr.py \
      --DR dataset \
      --ckpt_path ckpts/R1c_jayin_140ep/epoch=135.ckpt \
      --out_dir figures/perframe_R1c_ep135 \
      --z_scale 0.5 --seed 42
"""
import argparse, os, csv, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Make Trainer importable for VAE_Model + Generate_PSNR
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Trainer import VAE_Model, Generate_PSNR  # type: ignore
from dataloader import Dataset_Dance  # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--DR', type=str, default='dataset')
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='figures/perframe')
    p.add_argument('--z_scale', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_vi_len', type=int, default=630)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    # Architecture defaults — must match training
    p.add_argument('--F_dim', type=int, default=128)
    p.add_argument('--L_dim', type=int, default=32)
    p.add_argument('--N_dim', type=int, default=12)
    p.add_argument('--D_out_dim', type=int, default=192)
    p.add_argument('--frame_H', type=int, default=32)
    p.add_argument('--frame_W', type=int, default=64)
    # Stubs needed by VAE_Model.__init__
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_scheduler', type=str, default='cosine')
    p.add_argument('--lr_min', type=float, default=0.0)
    p.add_argument('--num_epoch', type=int, default=70)
    p.add_argument('--kl_anneal_type', type=str, default='Cyclical')
    p.add_argument('--kl_anneal_cycle', type=int, default=10)
    p.add_argument('--kl_anneal_ratio', type=float, default=0.5)
    p.add_argument('--tfr', type=float, default=1.0)
    p.add_argument('--tfr_d_step', type=float, default=0.1)
    p.add_argument('--tfr_sde', type=int, default=10)
    p.add_argument('--train_vi_len', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--save_root', type=str, default='/tmp/_unused')
    p.add_argument('--store_visualization', action='store_true')
    p.add_argument('--fast_train', action='store_true')
    p.add_argument('--fast_partial', type=float, default=1.0)
    p.add_argument('--fast_train_epoch', type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def per_frame_psnr(model, img, label, z_scale, device):
    img = img.to(device).permute(1, 0, 2, 3, 4)       # (T,B,C,H,W)
    label = label.to(device).permute(1, 0, 2, 3, 4)
    T, B, C, H, W = img.shape
    psnrs = []
    prev = img[0]
    for t in range(1, T):
        cur_gt = img[t]
        cur_pose = label[t]
        enc_prev = model.frame_transformation(prev)
        enc_pose = model.label_transformation(cur_pose)
        if z_scale == 0.0:
            z = torch.zeros(B, model.args.N_dim, enc_prev.shape[-2], enc_prev.shape[-1],
                            device=device, dtype=enc_prev.dtype)
        else:
            z = torch.randn(B, model.args.N_dim, enc_prev.shape[-2], enc_prev.shape[-1],
                            device=device, dtype=enc_prev.dtype) * z_scale
        fused = model.Decoder_Fusion(enc_prev, enc_pose, z)
        pred = model.Generator(fused)
        pred = pred.clamp(0, 1)
        pred = torch.nan_to_num(pred, nan=0.5)
        psnrs.append(float(Generate_PSNR(pred, cur_gt).detach().cpu()))
        prev = pred
    return psnrs


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = VAE_Model(args).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.frame_H, args.frame_W)),
        transforms.ToTensor(),
    ])
    dset = Dataset_Dance(root=args.DR, transform=transform, mode='val',
                         video_len=args.val_vi_len, partial=1.0)
    loader = DataLoader(dset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    all_psnrs = []  # list per sequence
    for i, (img, label) in enumerate(loader):
        psnrs = per_frame_psnr(model, img, label, args.z_scale, device)
        all_psnrs.append(psnrs)
        print(f"[seq {i}] mean PSNR = {np.mean(psnrs):.3f} dB  (T-1 = {len(psnrs)})")

    # All val seqs assumed same length
    arr = np.array(all_psnrs)  # (N_seq, T-1)
    mean_curve = arr.mean(axis=0)
    frame_idx = np.arange(1, arr.shape[1] + 1)

    # CSV
    csv_path = os.path.join(args.out_dir, 'per_frame_psnr.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_idx', 'mean_psnr_db'] + [f'seq{i}' for i in range(arr.shape[0])])
        for k in range(arr.shape[1]):
            w.writerow([int(frame_idx[k]), float(mean_curve[k])] + [float(arr[i, k]) for i in range(arr.shape[0])])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    label = 'PSNR' if arr.shape[0] == 1 else f'mean over {arr.shape[0]} seq'
    ax.plot(frame_idx, mean_curve, color='tab:red', lw=1.5, label=label)
    ax.set_xlabel('Frame index')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'Per-frame PSNR — {os.path.basename(args.ckpt_path)} '
                 f'(z_scale={args.z_scale}, mean={mean_curve.mean():.2f} dB)')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, 'per_frame_psnr.png')
    fig.savefig(png_path, dpi=150)
    print(f"saved: {png_path}\nsaved: {csv_path}")
    print(f"overall mean PSNR = {mean_curve.mean():.3f} dB")


if __name__ == '__main__':
    main()
