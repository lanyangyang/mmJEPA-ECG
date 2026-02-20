"""
mmJEPA Pre-Training Script

Key Features:
- Student-Teacher architecture with EMA updates
- Multi-mask prediction (time, channel, block masks)
# - Heart rate prediction
- Cosine learning rate scheduling with warmup
- Weight decay scheduling
- Wandb logging support

Usage:
    python mmJEPA.py -epochs 300 -batch_size 128 -lr 5e-4

Loss Components:
    - Time mask loss
    - Channel mask loss
    - Block mask loss
#    - Heart rate loss
"""

import argparse
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
import math
from datetime import datetime
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.utils import clip_grad_norm_
from block.data_load import npyLoad
from block.backbone_multi_target import context_encoder, attn_MultiMaskPredictor
from block.utils import build_wd_scheduler, build_ema_scheduler, show_masks
from block.block_mask import mm_mask

import time, contextlib, itertools

torch.set_float32_matmul_precision('high')

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with: pip install thop")

try:
    import subprocess
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False

@contextlib.contextmanager
def timer(msg):
    torch.cuda.synchronize()
    t0 = time.time()
    yield
    torch.cuda.synchronize()
    print(f"{msg}: {time.time() - t0:.3f}s")

def get_gpu_power():

    if not NVIDIA_SMI_AVAILABLE:
        return None
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return None

def compute_model_flops(student, teacher, predictor, rcg, pos, mask_list, device):

    if not THOP_AVAILABLE:
        return None, None

    try:

        x_obs = torch.ones_like(rcg[:1])
        pos_sample = pos[:1]

        with torch.no_grad():
            student_flops, _ = profile(student, inputs=(rcg[:1], pos_sample, x_obs), verbose=False)

            _, x_pad = student(rcg[:1], position=pos_sample, obs_m=x_obs)

            mt, mc, mb = mask_list
            mask_list_sample = (mt[:1], mc[:1], mb[:1])
            predictor_flops, _ = profile(predictor, inputs=(x_pad, mask_list_sample), verbose=False)

            teacher_flops, _ = profile(teacher, inputs=(rcg[:1], pos_sample, None), verbose=False)

        total_flops = student_flops + predictor_flops + teacher_flops
        return total_flops, (student_flops, predictor_flops, teacher_flops)
    except Exception as e:
        print(f"FLOPs computation failed: {e}")
        return None, None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-mask_ch', type=int, default=10)
    parser.add_argument('-mask_time', type=int, default=2)
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-lr', type=float, default=5e-4)
    parser.add_argument('-ema_m', type=float, default=0.996)
    parser.add_argument('-d_model', type=int, default=192)
    parser.add_argument('-n_vars', type=int, default=50)
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-pre_layers', type=int, default=1)
    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-d_ff', type=int, default=256)
    parser.add_argument('-ema_warmup_ratio', type=float, default=0.1)
    parser.add_argument('-eta_min_ratio', type=float, default=0.2)
    parser.add_argument('-warm', type=float, default=0.1)
    parser.add_argument('-patch_size', type=int, default=32)
    parser.add_argument('-stride', type=int, default=32)
    parser.add_argument('-use_wandb', action='store_true')
    parser.add_argument('-device', default='cuda:7')

    parser.add_argument('-profile_epoch', type=int, default=1)
    parser.add_argument('-track_power', action='store_true')

    args = parser.parse_args()
    return args

args = parse_args()
batch_size = args.batch_size
patch_size = args.patch_size
mask_ch = args.mask_ch
mask_time = args.mask_time
stride = args.stride
num_epochs = args.epochs
lr = args.lr
ema_m = args.ema_m
d_model = args.d_model
num_patch = 1 + (512 - args.patch_size) // args.stride
n_vars = args.n_vars
num_workers = args.num_workers
n_layers = args.n_layers
pre_layers = args.pre_layers
n_heads = args.n_heads
d_ff = args.d_ff
eta_min_ratio = args.eta_min_ratio

torch.manual_seed(42)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
param_str = f"lr{lr:.0e}_"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"checkpoints/{param_str}_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

stats_log_path = os.path.join(save_dir, "statistics.txt")
stats_file = open(stats_log_path, 'w', encoding='utf-8')

def write_to_stats(message):

    stats_file.write(message + '\n')
    stats_file.flush()

train_ds = npyLoad("./data/train_UD_RMS_all.npy", device='cpu')
test_ds  = npyLoad('./data/test_UD_RMS_all.npy',  device='cpu')

dl_kwargs = dict(batch_size=batch_size, num_workers=8,
                 pin_memory=True, persistent_workers=True)

train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kwargs)
test_loader = DataLoader(test_ds, shuffle=False, drop_last=True, **dl_kwargs)

student = context_encoder(
    c_in=n_vars, num_patch=num_patch, patch_len=patch_size,
    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dp=0.2,
    device=device).to(device)

teacher = deepcopy(student).eval().to(device)
for p in teacher.parameters():
    p.requires_grad_(False)

predictor = attn_MultiMaskPredictor(
    d_model=d_model, n_heads=n_heads // 2,
    d_ff=d_ff // 4, dropout=0.1).to(device)

wd_start = 5e-4
wd_end = 4e-3
wd_iter = build_wd_scheduler(wd_start, wd_end, int(num_epochs * len(train_loader) * 1.25),
                             num_epochs * len(train_loader))

optim = torch.optim.AdamW(
    list(student.parameters()) + list(predictor.parameters()),
    lr=lr, weight_decay=wd_start
)

criterion = nn.L1Loss()

total_steps = len(train_loader) * num_epochs
warm_steps = int(args.warm * total_steps)

def lr_lambda(step):
    if step < warm_steps:
        return step / warm_steps
    progress = (step - warm_steps) / (total_steps - warm_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return eta_min_ratio + (1 - eta_min_ratio) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

steps_per_ep = len(train_loader)

ema_iter = build_ema_scheduler(
    ema_momentum=ema_m,
    num_epochs=int(num_epochs * 1.25),
    steps_per_ep=steps_per_ep,
    warmup_steps=warm_steps
)

def ema_update(ema_net: torch.nn.Module,
               net: torch.nn.Module,
               m: float) -> None:
    with torch.no_grad():
        p_ema, p = list(ema_net.parameters()), list(net.parameters())
        torch._foreach_mul_(p_ema, m)
        torch._foreach_add_(p_ema, p, alpha=1.0 - m)

if args.use_wandb:
    config = {k: str(v) for k, v in vars(args).items()}
    run = wandb.init(
        project="mmECG-JEPA-multi-mask",
        name=param_str,
        config=config,
    )

write_to_stats(f"\n{'='*60}")
write_to_stats("Model Statistics")
write_to_stats(f"{'='*60}")
student_params = sum(p.numel() for p in student.parameters())
teacher_params = sum(p.numel() for p in teacher.parameters())
predictor_params = sum(p.numel() for p in predictor.parameters())
trainable_params = student_params + predictor_params
total_params = student_params + teacher_params + predictor_params
write_to_stats(f"Student:   {student_params:>10,} ({student_params/1e6:.2f}M)")
write_to_stats(f"Teacher:   {teacher_params:>10,} ({teacher_params/1e6:.2f}M)")
write_to_stats(f"Predictor: {predictor_params:>10,} ({predictor_params/1e6:.2f}M)")
write_to_stats(f"Trainable: {trainable_params:>10,} ({trainable_params/1e6:.2f}M)")
write_to_stats(f"Total:     {total_params:>10,} ({total_params/1e6:.2f}M)")
write_to_stats(f"{'='*60}\n")

flops_computed = False
total_flops = None

for epoch in range(1, num_epochs + 1):
    with timer("Epoch total train/test"):
        student.train()
        predictor.train()

        train_student_norm_mean_sum = 0.0
        train_student_norm_std_sum = 0.0
        train_teacher_norm_mean_sum = 0.0
        train_teacher_norm_std_sum = 0.0
        train_norm_cnt = 0

        train_loss_sum = 0.0
        train_time_loss_sum = 0.0
        train_cha_loss_sum = 0.0
        train_blk_loss_sum = 0.0

        n_train_batches = 0

        batch_latencies = []
        batch_powers = []
        profile_this_epoch = (epoch == args.profile_epoch)

        for batch_idx, (rcg, ecg, pos, y) in enumerate(train_loader):

            if profile_this_epoch:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_start_time = time.time()
                if args.track_power:
                    power_before = get_gpu_power()

            rcg = rcg.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)

            B, L, V, _ = rcg.shape
            D = d_model

            x_obs, mt, mc, mb = mm_mask(rcg, obs=[16, 8], time=mask_time, ch=mask_ch, tc=[8, 8])
            x_ctx, x_pad = student(rcg, position=pos, obs_m=x_obs)

            if profile_this_epoch and not flops_computed and batch_idx == 0:
                write_to_stats(f"\nComputing FLOPs for epoch {epoch}...")
                mask_list = (mt, mc, mb)
                total_flops, flops_breakdown = compute_model_flops(
                    student, teacher, predictor, rcg, pos, mask_list, device
                )
                if total_flops is not None:
                    flops_computed = True
                    write_to_stats(f"Total FLOPs per sample: {total_flops/1e9:.2f} GFLOPs")
                    if flops_breakdown:
                        student_flops, predictor_flops, teacher_flops = flops_breakdown
                        write_to_stats(f"  Student: {student_flops/1e9:.2f} GFLOPs")
                        write_to_stats(f"  Predictor: {predictor_flops/1e9:.2f} GFLOPs")
                        write_to_stats(f"  Teacher: {teacher_flops/1e9:.2f} GFLOPs")
                    write_to_stats(f"Total FLOPs per batch (B={B}): {total_flops*B/1e9:.2f} GFLOPs\n")

            student_feat_norms = x_ctx.norm(dim=-1)
            batch_stu_mean = student_feat_norms.mean().item()
            batch_stu_std = student_feat_norms.std().item()

            train_student_norm_mean_sum += batch_stu_mean
            train_student_norm_std_sum += batch_stu_std

            mask_list = (mt, mc, mb)
            hat1, hat2, hat3 = predictor(x_pad, mask_list)[:3]

            with torch.no_grad():
                x_full_ctx = teacher(rcg, position=pos, obs_m=None)
                teacher_feat_norms = x_full_ctx.norm(dim=-1)
                batch_tea_mean = teacher_feat_norms.mean().item()
                batch_tea_std = teacher_feat_norms.std().item()
                train_teacher_norm_mean_sum += batch_tea_mean
                train_teacher_norm_std_sum += batch_tea_std

                yT1 = x_full_ctx[mt].view(B, -1, D)
                yT2 = x_full_ctx[mc].view(B, -1, D)
                yT3 = x_full_ctx[mb].view(B, -1, D)

            time_loss = criterion(hat1, yT1)
            cha_loss = criterion(hat2, yT2)
            blk_loss = criterion(hat3, yT3)

            loss = time_loss + cha_loss + blk_loss

            wd_cur = next(wd_iter)
            for pg in optim.param_groups:
                pg['weight_decay'] = wd_cur

            optim.zero_grad()
            loss.backward()

            clip_grad_norm_(
                list(student.parameters()) + list(predictor.parameters()),
                max_norm=1.0
            )

            optim.step()
            scheduler.step()

            m_cur = next(ema_iter)
            ema_update(teacher, student, m_cur)

            train_loss_sum += loss.item()
            train_time_loss_sum += time_loss.item()
            train_cha_loss_sum += cha_loss.item()
            train_blk_loss_sum += blk_loss.item()

            n_train_batches += 1

            train_norm_cnt += 1

            if profile_this_epoch:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_end_time = time.time()
                batch_latency = batch_end_time - batch_start_time
                batch_latencies.append(batch_latency)

                if args.track_power:
                    power_after = get_gpu_power()
                    if power_before is not None and power_after is not None:
                        avg_power = (power_before + power_after) / 2
                        batch_powers.append(avg_power)

        avg_time_loss = train_time_loss_sum / n_train_batches
        avg_cha_loss = train_cha_loss_sum / n_train_batches
        avg_blk_loss = train_blk_loss_sum / n_train_batches

        student.eval()
        predictor.eval()

        test_loss_sum = 0.0
        test_time_loss_sum = 0.0
        test_cha_loss_sum = 0.0
        test_blk_loss_sum = 0.0

        n_test_batches = 0

        with torch.no_grad():
            for batch_idx, (rcg, ecg, pos, y) in enumerate(test_loader):
                rcg = rcg.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)

                B, L, V, _ = rcg.shape
                D = d_model

                x_obs_m, mt, mc, mb = mm_mask(
                    rcg,
                    obs=[16, 8],
                    time=mask_time,
                    ch=mask_ch,
                    tc=[8, 8]
                )

                _, z_pad = student(rcg, position=pos, obs_m=x_obs_m)

                hat1, hat2, hat3 = predictor(z_pad, (mt, mc, mb))[:3]

                x_full_ctx = teacher(rcg, position=pos, obs_m=None)

                yT1 = x_full_ctx[mt].view(B, -1, D)
                yT2 = x_full_ctx[mc].view(B, -1, D)
                yT3 = x_full_ctx[mb].view(B, -1, D)

                time_loss = criterion(hat1, yT1)
                cha_loss = criterion(hat2, yT2)
                blk_loss = criterion(hat3, yT3)

                loss = time_loss + cha_loss + blk_loss

                test_loss_sum += loss.item()
                test_time_loss_sum += time_loss.item()
                test_cha_loss_sum += cha_loss.item()
                test_blk_loss_sum += blk_loss.item()

                n_test_batches += 1

        test_avg_time_loss = test_time_loss_sum / n_test_batches
        test_avg_cha_loss = test_cha_loss_sum / n_test_batches
        test_avg_blk_loss = test_blk_loss_sum / n_test_batches

        avg_stu_mean = train_student_norm_mean_sum / train_norm_cnt
        avg_stu_std = train_student_norm_std_sum / train_norm_cnt
        avg_tea_mean = train_teacher_norm_mean_sum / train_norm_cnt
        avg_tea_std = train_teacher_norm_std_sum / train_norm_cnt

        print(f"Epoch {epoch:02d} | "
              f"tr_time_loss={avg_time_loss:.5f} | "
              f"tr_cha_loss={avg_cha_loss:.5f} | "
              f"tr_blk_loss={avg_blk_loss:.5f} | "
              f"tr_total_loss={avg_time_loss + avg_cha_loss + avg_blk_loss:.5f}")

        print(f"Epoch {epoch:02d} | "
              f"te_time_loss={test_avg_time_loss:.5f} | "
              f"te_cha_loss={test_avg_cha_loss:.5f} | "
              f"te_blk_loss={test_avg_blk_loss:.5f} | "
              f"te_total_loss={test_avg_time_loss + test_avg_cha_loss + test_avg_blk_loss:.5f}")

        if profile_this_epoch and len(batch_latencies) > 0:
            write_to_stats(f"\n{'='*60}")
            write_to_stats(f"Profiling Statistics (Epoch {epoch})")
            write_to_stats(f"{'='*60}")

            avg_latency = sum(batch_latencies) / len(batch_latencies)
            min_latency = min(batch_latencies)
            max_latency = max(batch_latencies)
            std_latency = (sum((t - avg_latency) ** 2 for t in batch_latencies) / len(batch_latencies)) ** 0.5

            write_to_stats(f"Latency per batch (B={batch_size}):")
            write_to_stats(f"  Average: {avg_latency*1000:.2f} ms")
            write_to_stats(f"  Min:     {min_latency*1000:.2f} ms")
            write_to_stats(f"  Max:     {max_latency*1000:.2f} ms")
            write_to_stats(f"  Std:     {std_latency*1000:.2f} ms")
            write_to_stats(f"  Throughput: {batch_size/avg_latency:.2f} samples/sec")
            write_to_stats(f"  Per sample: {avg_latency/batch_size*1000:.2f} ms")

            if len(batch_powers) > 0:
                avg_power = sum(batch_powers) / len(batch_powers)
                min_power = min(batch_powers)
                max_power = max(batch_powers)
                energy_per_batch = avg_power * avg_latency

                write_to_stats(f"\nPower Consumption:")
                write_to_stats(f"  Average: {avg_power:.2f} W")
                write_to_stats(f"  Min:     {min_power:.2f} W")
                write_to_stats(f"  Max:     {max_power:.2f} W")
                write_to_stats(f"  Energy per batch: {energy_per_batch:.4f} J")
                write_to_stats(f"  Energy per sample: {energy_per_batch/batch_size:.4f} J")

            if total_flops is not None:
                write_to_stats(f"\nComputational Complexity:")
                write_to_stats(f"  FLOPs per sample: {total_flops/1e9:.2f} GFLOPs")
                write_to_stats(f"  FLOPs per batch:  {total_flops*batch_size/1e9:.2f} GFLOPs")

                if len(batch_powers) > 0:
                    gflops_per_watt = (total_flops * batch_size / 1e9) / (avg_power * avg_latency) if avg_power > 0 else 0
                    write_to_stats(f"  Efficiency: {gflops_per_watt:.2f} GFLOPs/J")

            write_to_stats(f"{'='*60}\n")

        if args.use_wandb:
            log_dict = {
                "epoch": epoch,

                "train/avg_time_loss": avg_time_loss,
                "train/avg_cha_loss": avg_cha_loss,
                "train/avg_blk_loss": avg_blk_loss,

                "train/avg_total": avg_time_loss + avg_cha_loss + avg_blk_loss,
                "train/student_norm_mean": avg_stu_mean,
                "train/student_norm_std": avg_stu_std,
                "train/teacher_norm_mean": avg_tea_mean,
                "train/teacher_norm_std": avg_tea_std,

                "val/avg_time_loss": test_avg_time_loss,
                "val/avg_cha_loss": test_avg_cha_loss,
                "val/avg_blk_loss": test_avg_blk_loss,

                "val/avg_total": test_avg_time_loss + test_avg_cha_loss + test_avg_blk_loss,

                "lr": scheduler.get_last_lr()[0],
            }

            if profile_this_epoch and len(batch_latencies) > 0:
                log_dict["profile/avg_latency_ms"] = avg_latency * 1000
                log_dict["profile/throughput_samples_per_sec"] = batch_size / avg_latency
                log_dict["profile/per_sample_latency_ms"] = avg_latency / batch_size * 1000

                if len(batch_powers) > 0:
                    log_dict["profile/avg_power_watts"] = avg_power
                    log_dict["profile/energy_per_batch_joules"] = energy_per_batch
                    log_dict["profile/energy_per_sample_joules"] = energy_per_batch / batch_size

                if total_flops is not None:
                    log_dict["profile/flops_per_sample_gflops"] = total_flops / 1e9
                    log_dict["profile/flops_per_batch_gflops"] = total_flops * batch_size / 1e9
                    if len(batch_powers) > 0 and avg_power > 0:
                        log_dict["profile/efficiency_gflops_per_joule"] = (total_flops * batch_size / 1e9) / (avg_power * avg_latency)

            wandb.log(log_dict, step=epoch)

        save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)

stats_file.close()
print(f"\nStatistics saved to: {stats_log_path}")

wandb.finish()
