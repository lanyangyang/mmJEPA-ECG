import math

import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def mm_mask(x,
                     obs=(12, 8),          # Obs_H, Obs_W
                     time=1,               # T1_W
                     ch=8,                 # T2_W
                     tc=(8, 8),            # T3_H, T3_W
                     max_retry=20):
    """
    Returns: obs_m, t1_m, t2_m, t3_m (Bool) shape [B,L,V]
    Only ensures that the three blocks do not overlap with Obs; block sizes are fixed.
    """
    B, L, V, _ = x.shape
    dev = x.device

    OBS_H, OBS_W = obs
    T1_W         = time
    T2_H, T2_W   = 1, ch
    T3_H, T3_W   = tc

    obs_m = torch.zeros(B,L,V, dtype=torch.bool, device=dev)
    t1_m  = torch.zeros_like(obs_m)
    t2_m  = torch.zeros_like(obs_m)
    t3_m  = torch.zeros_like(obs_m)

    for b in range(B):
        # ------------ Resample Obs & T1 & T2 until feasible ------------
        for _ in range(max_retry):
            # --- Obs ---
            top  = random.randint(0, L-OBS_H)
            left = random.randint(0, V-OBS_W)

            # Available columns = non-Obs columns
            free_cols = list(set(range(V)) - set(range(left, left+OBS_W)))
            if len(free_cols) < (T1_W + T2_W):
                continue           

            # --- T1 full column ---
            valid_cols = [c for c in free_cols if c + T1_W <= V]
            if len(valid_cols) == 0:
                continue  # No valid T1 starting point, resample Obs
            col_t1 = random.choice(valid_cols)
            free_cols_no_t1 = list(set(free_cols) - set(range(col_t1, col_t1+T1_W)))
            if len(free_cols_no_t1) < T2_W:
                continue            # Check again, if still insufficient continue retry

            # --- T2 (1×ch) ---
            row_t2 = random.randint(0, L - T2_H)
            cols_t2 = random.sample(free_cols_no_t1, T2_W)

            # ---------- Write all ----------
            obs_m[b, top:top+OBS_H, left:left+OBS_W] = True
            t1_m[b, :, col_t1:col_t1+T1_W]           = True
            t2_m[b, row_t2:row_t2+T2_H, cols_t2]     = True
            break
        else:
            raise RuntimeError("multi_mask_fixed: Obs/T1/T2 retry limit exceeded, please reduce sizes")

        # ------------ T3 (tc) ------------
        for _ in range(max_retry):
            r0 = random.randint(0, L-T3_H)
            c0 = random.randint(0, V-T3_W)
            if not obs_m[b, r0:r0+T3_H, c0:c0+T3_W].any():
                t3_m[b, r0:r0+T3_H, c0:c0+T3_W] = True
                break
        else:
            raise RuntimeError("multi_mask_fixed: T3 retry limit exceeded, please reduce tc size")

    return obs_m, t1_m, t2_m, t3_m

# ----------- (Optional) Convert mask to token sequence -----------------
def gather_tokens(x: torch.Tensor, mask: torch.Tensor):
    """
    x : [B,L,V,D]   mask : [B,L,V] (True = extract)
    return : [B, N_token, D] (same number of tokens per sample ⇒ fixed here)
    """
    B, L, V, D = x.shape
    tokens = []
    for b in range(B):
        tok = x[b][mask[b]].view(-1, D)   # [N, D]
        tokens.append(tok)
    return torch.stack(tokens)            # Stack to [B, N, D]


def create_patch_batch(xb, patch_len, stride):
    """
    xb: [n_segment, seq_len, n_vars] (or n_vars=1 for ECG)
    return: [n_segment, num_patch, patch_len, n_vars]
    """
    n_seg, seq_len, n_vars = xb.shape
    num_patch = 1 + (seq_len - patch_len) // stride
    out = np.empty((n_seg, num_patch, patch_len, n_vars), dtype=xb.dtype)
    for i in range(n_seg):
        for j in range(num_patch):
            start = j * stride
            out[i, j] = xb[i, start: start + patch_len, :]
    return out
