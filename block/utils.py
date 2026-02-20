"""
Utility functions
"""

import torch
from torch import nn
import math
import numpy as np
from typing import Iterator
from collections import OrderedDict
import matplotlib.pyplot as plt

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

def build_ema_scheduler(
        ema_momentum: float,
        num_epochs: int,
        steps_per_ep: int,
        warmup_steps: int = 0
):
    total_steps = num_epochs * steps_per_ep

    def _gen():
        for i in range(total_steps + 1):
            if i < warmup_steps:
                yield ema_momentum
            else:
                progress = (i - warmup_steps) / max(1, total_steps - warmup_steps)
                m = ema_momentum + progress * (1.0 - ema_momentum)
                yield m

    return _gen()

def build_wd_scheduler(start, end, total_steps, train_steps):
    def _gen():
        for i in range(train_steps):
            t = i / max(1, total_steps - 1)
            yield start + t * (end - start)

    return _gen()

def strip_prefix_from_state_dict(state_dict, prefixes):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        for p in prefixes:
            if new_key.startswith(p):
                new_key = new_key[len(p):]
                break
        new_state_dict[new_key] = v
    return new_state_dict

def cosine_noise_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    min_alpha_cumprod = 1e-4
    alphas_cumprod = np.clip(alphas_cumprod, a_min=min_alpha_cumprod, a_max=1.0)

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas

def show_masks(mt, mc, mb, title="mask vis"):

    m_names = ["time", "channel", "block"]
    masks   = [mt[0].cpu().numpy(), mc[0].cpu().numpy(), mb[0].cpu().numpy()]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    for ax, m, name in zip(axs, masks, m_names):
        ax.imshow(m, cmap="gray_r", interpolation="nearest", aspect="auto")
        ax.set_title(name); ax.set_xlabel("V"); ax.set_ylabel("L")
    fig.suptitle(title); plt.tight_layout(); plt.show()
