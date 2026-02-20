"""
ECG Generation Training Script

This script trains a diffusion model to generate ECG signals from RCG (radar) data.
It combines a pretrained encoder with a diffusion model for end-to-end training.

Key features:
- Uses pretrained encoder weights from JEPA training
- Supports frequency domain loss
- Supports fine-tuning mode
"""

import argparse
import os
import math
import logging
from glob import glob
from time import time
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from custom_model import HRC_dit
from diffusion import create_diffusion
from block.data_load import npyLoad
from block.backbone_multi_target import context_encoder
from block.Positionemb import build_2d_sincos_pe

def frequency_domain_loss(pred_ecg, target_ecg):
    pred_fft = torch.fft.fft(pred_ecg, dim=-1)
    target_fft = torch.fft.fft(target_ecg, dim=-1)

    magnitude_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
    phase_loss     = F.l1_loss(torch.angle(pred_fft), torch.angle(target_fft))

    return magnitude_loss + 0.5 * phase_loss

def patch_ecg(sig: torch.Tensor) -> torch.Tensor:

    if sig.dim() == 3 and sig.shape[-1] == 1:
        sig = sig.squeeze(-1)
    B = sig.shape[0]
    if sig.shape[1] < 576:
        sig = F.pad(sig, (0, 576 - sig.shape[1]))
    return sig.reshape(B, 1, 24, 24)

def reorder_rcg(x: torch.Tensor, n_vars: int) -> torch.Tensor:

    if x.dim() != 4:
        raise ValueError(f"RCG tensor must be 4-D, got {x.dim()}-D")
    B, L = x.shape[:2]
    if x.shape[2] == n_vars:
        return x
    elif x.shape[3] == n_vars:
        return x.permute(0, 1, 3, 2).contiguous()
    else:
        raise RuntimeError("Unable to infer RCG layout. Please check data dimensions.")

def ecg_patches_to_full(ecg_p: torch.Tensor, patch_len: int) -> torch.Tensor:

    if ecg_p.dim() != 4:
        raise ValueError("ECG patch tensor must be 4-D")
    if ecg_p.shape[2] == patch_len:
        ecg_full = ecg_p.reshape(ecg_p.size(0), -1, 1)
    elif ecg_p.shape[3] == patch_len:
        ecg_full = ecg_p.permute(0,1,3,2).reshape(ecg_p.size(0), -1, 1)
    else:
        raise RuntimeError("Unrecognized ECG patch layout")
    return ecg_full

@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999):

    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)

def requires_grad(m: torch.nn.Module, flag: bool = True):
    for p in m.parameters():
        p.requires_grad_(flag)

def plot_ecg_pairs(gt: np.ndarray, pred: np.ndarray, save_path: str):

    assert gt.shape == pred.shape, "Ground truth and prediction shape mismatch"
    N = gt.shape[0]
    fig, axes = plt.subplots(N, 2, figsize=(8, 2*N))
    axes[0][0].set_title('ECG GroundTruth')
    axes[0][1].set_title('Generated ECG')
    for i in range(N):
        axes[i][0].plot(gt[i], color='m')
        axes[i][1].plot(pred[i], color='y')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def main(args: argparse.Namespace):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    os.makedirs(args.results_dir, exist_ok=True)
    exp_idx = len(glob(f"{args.results_dir}/*"))
    exp_dir = f"{args.results_dir}/{exp_idx:03d}-e2e"
    ckpt_dir = f"{exp_dir}/checkpoints";  os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{exp_dir}/log.txt")])
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment dir: {exp_dir}")

    num_patch = 1 + (512 - args.patch_len) // args.stride
    student = context_encoder(
        c_in=args.n_vars,
        num_patch=num_patch,
        patch_len=args.patch_len,
        d_model=args.enc_dim,
        n_layers=args.enc_layers,
        n_heads=args.enc_heads,
        d_ff=args.enc_ff,
        dp=0,
        device=device
    ).to(device)

    model = HRC_dit(learn_sigma=False).to(device)
    ema   = deepcopy(model).to(device);  requires_grad(ema, False)

    student = torch.compile(student, backend="inductor", mode="default")
    model   = torch.compile(model,   backend="inductor", mode="default")
    ema     = torch.compile(ema,     backend="inductor", mode="default")
    logger.info("Enabled torch.compile")

    if args.encoder_ckpt and os.path.isfile(args.encoder_ckpt):
        state_dict = torch.load(args.encoder_ckpt, map_location=device)['student_state_dict']

        w_pos_key = None
        if '_orig_mod.W_pos' in state_dict:
            w_pos_key = '_orig_mod.W_pos'
        elif 'W_pos' in state_dict:
            w_pos_key = 'W_pos'

        if w_pos_key:
            trained_vars = state_dict[w_pos_key].shape[1]
            current_vars = args.n_vars
            L = state_dict[w_pos_key].shape[0]
            D = state_dict[w_pos_key].shape[2]

            if trained_vars != current_vars:
                logger.info(f"Detected channel mismatch: ckpt {trained_vars} vs current {current_vars}. Regenerating PE …")

                del state_dict[w_pos_key]
                student.load_state_dict(state_dict, strict=False)

                new_pe = build_2d_sincos_pe(L, current_vars, D, device)
                target_mod = student._orig_mod if hasattr(student, '_orig_mod') else student
                target_mod.register_buffer('W_pos', new_pe)
                logger.info(f"Registered new PE with shape {tuple(new_pe.shape)}")
            else:
                missing, unexpected = student.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"Encoder missing keys: {missing}")
                if unexpected:
                    logger.warning(f"Encoder unexpected keys: {unexpected}")
        else:
            missing, unexpected = student.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Encoder missing keys: {missing}")
            if unexpected:
                logger.warning(f"Encoder unexpected keys: {unexpected}")
        logger.info("Loaded encoder weights.")

    diffusion = create_diffusion(timestep_respacing="", learn_sigma=False)

    base_lr = args.lr
    if args.finetune:
        requires_grad(student, False)

        last_layer = list(student.parameters())[-1]
        last_layer.requires_grad = True
        encoder_params = [p for n, p in student.named_parameters() if p.requires_grad]
        other_params = list(model.parameters())

        opt = torch.optim.AdamW([
            {'params': encoder_params, 'lr': base_lr * 0.1},
            {'params': other_params, 'lr': base_lr}
        ], weight_decay=1e-4)
    else:
        requires_grad(student, False)
        student.eval()
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=1e-4)

    train_ds = npyLoad(args.train_npy, device='cpu')
    test_ds  = npyLoad(args.test_npy,  device='cpu')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,shuffle=True, num_workers=8,
                                pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=8,
                                pin_memory=True, persistent_workers=True)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_steps, total_steps - 1)
    min_lr = 1e-6

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cosine * (1 - min_lr / base_lr) + (min_lr / base_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    logger.info(f"Start training for {args.epochs} epochs …")
    running_loss, log_steps = 0.0, 0
    train_steps = 0
    t_start = time()

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        model.train()

        for rcg, ecg_p, pos, _ in tqdm(train_loader):
            rcg = rcg.to(device, non_blocking=True)
            ecg_p = ecg_p.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)

            rcg_fmt = reorder_rcg(rcg, n_vars=args.n_vars)
            up_feat = student(rcg_fmt, position=pos, obs_m=None)
            B = rcg_fmt.size(0)
            up_flat = up_feat.reshape(B, -1, up_feat.size(-1))

            ecg_full = ecg_patches_to_full(ecg_p, patch_len=args.patch_len)
            ecg_img  = patch_ecg(ecg_full)

            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

            noise = torch.randn_like(ecg_img)

            loss_dict = diffusion.training_losses(model, ecg_img, t,
                                                  model_kwargs={'upstream_output': up_flat},
                                                  noise=noise)
            ddpm_loss = loss_dict["loss"].mean()

            total_loss = ddpm_loss

            x_t = diffusion.q_sample(ecg_img, t, noise=noise)
            out = diffusion.p_mean_variance(
                model, x_t, t,
                model_kwargs={'upstream_output': up_flat}
            )
            pred_xstart = out["pred_xstart"]

            if args.use_freq_loss:
                pred_flat   = pred_xstart.reshape(B, -1)[..., :512]
                target_flat = ecg_img.reshape(B, -1)[..., :512]
                freq_loss  = frequency_domain_loss(pred_flat, target_flat)
                total_loss = total_loss + args.freq_loss_weight * freq_loss

            pred_ecg_512 = pred_xstart.reshape(-1, 576)[:, :512]
            target_ecg_512 = ecg_img.reshape(-1, 576)[:, :512]

            if args.recon_loss_type == 'l1':
                recon_loss = F.l1_loss(pred_ecg_512, target_ecg_512)
            else:
                recon_loss = F.mse_loss(pred_ecg_512, target_ecg_512)
            total_loss = total_loss + args.recon_loss_weight * recon_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()
            scheduler.step()
            update_ema(ema, model)

            running_loss += total_loss.item()
            log_steps += 1;  train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                t_now = time()
                logger.info(f"step={train_steps} | diff_loss={running_loss/log_steps:.4f} | "
                            f"lr={scheduler.get_last_lr()[0]:.2e} | "
                            f"{log_steps/(t_now-t_start):.2f} step/s")
                running_loss = 0.0;  log_steps = 0;  t_start = t_now

        if args.ckpt_every and (epoch+1) % args.ckpt_every == 0:
            ckpt_path = f"{ckpt_dir}/epoch_{epoch+1:03d}.pt"
            if args.finetune:
                torch.save({
                    "model": model.state_dict(),
                    "encoder": student.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args)
                }, ckpt_path)
            else:
                torch.save({
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args)
                }, ckpt_path)
            logger.info(f"Saved {ckpt_path}")

        if (epoch+1) % 30 == 0:
            evaluate(model, diffusion, student, test_loader, device, logger,
                     patch_len=args.patch_len, stride=args.stride,
                     results_dir=exp_dir, epoch=epoch+1, n_vars=args.n_vars)

    logger.info("Done.")

def evaluate(model, diffusion, encoder, loader, device, logger,
             patch_len:int, stride:int, results_dir:str, epoch:int, n_vars:int):

    model.eval();  encoder.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for rcg, ecg_p, pos, _ in loader:
            rcg = rcg.to(device, non_blocking=True)
            ecg_p = ecg_p.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)

            rcg_fmt = reorder_rcg(rcg, n_vars=n_vars)
            up_feat = encoder(rcg_fmt, position=pos, obs_m=None)

            B = rcg_fmt.size(0)
            up_flat = up_feat.reshape(B,-1,up_feat.size(-1))

            z = torch.randn(B,1,24,24, device=device)
            samples = diffusion.p_sample_loop(model, z.shape, z,
                                              clip_denoised=True,
                                              model_kwargs={'upstream_output': up_flat},
                                              device=device)
            ecg_full = ecg_patches_to_full(ecg_p, patch_len).to(device)

            gen_ecg_512 = samples.reshape(-1, 576)[:, :512]
            gt_ecg_512 = ecg_full[:, :512, 0]
            mse = F.mse_loss(gen_ecg_512, gt_ecg_512)
            total += mse.item()*B; n += B

            gt_np   = ecg_full[:7, :, 0].cpu().numpy()
            pred_np = samples[:7].reshape(7, -1)[:, :512].cpu().numpy()
            save_path = os.path.join(results_dir, f"epoch_{epoch:03d}_ecg.png")
            plot_ecg_pairs(gt_np, pred_np, save_path)

            break
    logger.info(f"Eval MSE at epoch {epoch}: {total/n:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('--train-npy', type=str, default='')  # change to your path
    ap.add_argument('--test-npy', type=str, default='')  # change to your path
    ap.add_argument('--n-vars', type=int, default=8)
    ap.add_argument('--patch-len', type=int, default=32)
    ap.add_argument('--stride', type=int, default=32)

    ap.add_argument('--encoder-ckpt', type=str, default='')  # change to your path
    ap.add_argument('--enc-dim', type=int, default=192)
    ap.add_argument('--enc-layers', type=int, default=2)
    ap.add_argument('--enc-heads', type=int, default=8)
    ap.add_argument('--enc-ff', type=int, default=256)

    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup-steps', type=int, default=7750)
    ap.add_argument('--log-every', type=int, default=155)
    ap.add_argument('--ckpt-every', type=int, default=50)
    ap.add_argument('--results-dir', type=str, default='')  # change to your path
    ap.add_argument('--finetune', action='store_true')

    ap.add_argument('--use-freq-loss', action='store_true')
    ap.add_argument('--freq-loss-weight', type=float, default=0.005)

    ap.add_argument('--recon-loss-type', type=str, default='l1', choices=['l1','mse'])
    ap.add_argument('--recon-loss-weight', type=float, default=0.1)

    ap.add_argument('--seed', type=int, default=42)

    args = ap.parse_args()
    main(args)
