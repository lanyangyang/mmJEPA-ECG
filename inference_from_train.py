"""
Inference script based on end2end_train.py
Maintains exact same data processing and model loading logic as training.
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    num_patch = 1 + (512 - args.patch_len) // args.stride
    student = context_encoder(
        c_in=args.n_vars,
        num_patch=num_patch,
        patch_len=args.patch_len,
        d_model=args.enc_dim,
        n_layers=args.enc_layers,
        n_heads=args.enc_heads,
        d_ff=args.enc_ff,
        dp=0.2,
        device=device
    ).to(device)

    model = HRC_dit(learn_sigma=False).to(device)

    student = torch.compile(student, backend="inductor", mode="default")
    model = torch.compile(model, backend="inductor", mode="default")
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
                logger.info(f"Detected channel mismatch: ckpt {trained_vars} vs current {current_vars}. Regenerating PE...")
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
    else:
        logger.warning("No encoder checkpoint found! Using randomly initialized encoder!")

    logger.info(f"Loading diffusion checkpoint from: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        logger.info("Loaded model weights")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded raw checkpoint")

    # DDIM sampling: 100 steps, deterministic (eta=0.0), ~10x faster than DDPM
    # To switch back to full DDPM (1000 steps): use timestep_respacing="" and p_sample_loop below
    diffusion = create_diffusion(timestep_respacing="100", learn_sigma=False)

    test_ds = npyLoad(args.test_npy, device='cpu')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8,
                            pin_memory=True, persistent_workers=True)

    model.eval()
    student.eval()

    logger.info("Starting inference on all test data...")
    all_mse = []
    all_generated_ecgs = []
    all_ground_truth_ecgs = []

    with torch.no_grad():
        for batch_idx, (rcg, ecg_p, pos, _) in enumerate(test_loader):
            rcg = rcg.to(device, non_blocking=True)
            ecg_p = ecg_p.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)

            rcg_fmt = reorder_rcg(rcg, n_vars=args.n_vars)
            up_feat = student(rcg_fmt, position=pos, obs_m=None)
            B = rcg_fmt.size(0)
            up_flat = up_feat.reshape(B, -1, up_feat.size(-1))

            if batch_idx == 0:
                logger.info(f"Debug shapes (first batch):")
                logger.info(f"  rcg: {rcg.shape} -> rcg_fmt: {rcg_fmt.shape}")
                logger.info(f"  up_feat: {up_feat.shape} -> up_flat: {up_flat.shape}")
                logger.info(f"  Expected up_flat: [B, 128, 192] = [{B}, {16*8}, 192]")
                logger.info(f"Debug statistics:")
                logger.info(f"  rcg range: [{rcg.min():.4f}, {rcg.max():.4f}], mean: {rcg.mean():.4f}")
                logger.info(f"  up_flat range: [{up_flat.min():.4f}, {up_flat.max():.4f}], mean: {up_flat.mean():.4f}")
                logger.info(f"  ecg_p range: [{ecg_p.min():.4f}, {ecg_p.max():.4f}], mean: {ecg_p.mean():.4f}")

            z = torch.randn(B, 1, 24, 24, device=device)
            # DDIM: deterministic, 100 steps
            samples = diffusion.ddim_sample_loop(model.forward, z.shape, z,
                                               clip_denoised=True,
                                               model_kwargs={'upstream_output': up_flat},
                                               device=device,
                                               eta=0.0)
            # DDPM (1000 steps, stochastic): uncomment to use instead
            # samples = diffusion.p_sample_loop(model.forward, z.shape, z,
            #                                   clip_denoised=True,
            #                                   model_kwargs={'upstream_output': up_flat},
            #                                   device=device)
            ecg_full = ecg_patches_to_full(ecg_p, patch_len=args.patch_len).to(device)

            if batch_idx == 0:
                logger.info(f"  samples range: [{samples.min():.4f}, {samples.max():.4f}], mean: {samples.mean():.4f}")
                logger.info(f"  ecg_full range: [{ecg_full.min():.4f}, {ecg_full.max():.4f}], mean: {ecg_full.mean():.4f}")

            gen_ecg_512 = samples.reshape(-1, 576)[:, :512]
            gt_ecg_512 = ecg_full[:, :512, 0]
            mse = F.mse_loss(gen_ecg_512, gt_ecg_512)

            all_mse.append(mse.item() * B)

            gen_ecg = gen_ecg_512.cpu().numpy()
            gt_ecg = gt_ecg_512.cpu().numpy()
            all_generated_ecgs.append(gen_ecg)
            all_ground_truth_ecgs.append(gt_ecg)

            if batch_idx == 0:
                gt_np = ecg_full[:7, :, 0].cpu().numpy()
                pred_np = samples[:7].reshape(7, -1)[:, :512].cpu().numpy()
                save_path = os.path.join(args.output_dir, f"inference_ecg.png")
                plot_ecg_pairs(gt_np, pred_np, save_path)
                logger.info(f"Saved visualization to {save_path}")

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                current_mse = mse.item()
                logger.info(f"Processed batch {batch_idx + 1}, current MSE: {current_mse:.6f}")

    final_generated_ecgs = np.concatenate(all_generated_ecgs, axis=0)
    final_ground_truth_ecgs = np.concatenate(all_ground_truth_ecgs, axis=0)

    np.save(os.path.join(args.output_dir, args.gen_ecg_name), final_generated_ecgs)
    np.save(os.path.join(args.output_dir, args.ecg_name), final_ground_truth_ecgs)

    total_mse = sum(all_mse)
    total_samples = final_generated_ecgs.shape[0]
    avg_mse = total_mse / total_samples
    total_batches = len(all_mse)

    logger.info(f"Inference completed!")
    logger.info(f"Processed {total_batches} batches with {total_samples} total samples")
    logger.info(f"Average MSE: {avg_mse:.6f} (total_mse={total_mse:.2f}, total_samples={total_samples})")
    logger.info(f"Generated ECG shape: {final_generated_ecgs.shape}")
    logger.info(f"Ground truth ECG shape: {final_ground_truth_ecgs.shape}")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('--test-npy', type=str, default='')  # change to your path
    ap.add_argument('--n-vars', type=int, default=8)
    ap.add_argument('--patch-len', type=int, default=32)
    ap.add_argument('--stride', type=int, default=32)

    ap.add_argument('--encoder-ckpt', type=str, default='')  # change to your path

    ap.add_argument('--enc-dim', type=int, default=192)
    ap.add_argument('--enc-layers', type=int, default=2)
    ap.add_argument('--enc-heads', type=int, default=8)
    ap.add_argument('--enc-ff', type=int, default=256)

    ap.add_argument('--ckpt', type=str, default='')  # change to your path

    ap.add_argument('--gpu-id', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--output-dir', type=str, default='')  # change to your path

    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--gen-ecg-name', type=str, default='gen_ecg_sim.npy')
    ap.add_argument('--ecg-name', type=str, default='ecg_sim.npy')


    args = ap.parse_args()
    main(args)
