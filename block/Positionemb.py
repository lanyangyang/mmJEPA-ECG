"""
Positional encoding utilities for 3D and 2D spatial data.
"""

import torch
from torch import nn
import math

def space_sincos_pe(xyz: torch.Tensor, d_model: int, normalize: bool = True):

    B, n_vars, C = xyz.shape
    assert C == 3, "Last dimension of xyz must be 3"
    assert d_model % 6 == 0, "d_model should be multiple of 6"
    dim_each = d_model // 3

    div_term = torch.exp(torch.arange(0, dim_each, 2, dtype=torch.float32, device=xyz.device) * -(math.log(10000.0) / dim_each))

    pe_list = []
    for i in range(3):
        pos = xyz[:, :, i]
        pos = pos.unsqueeze(-1)

        sin_term = torch.sin(pos * div_term)
        cos_term = torch.cos(pos * div_term)
        pe_list.append(sin_term)
        pe_list.append(cos_term)
    pe = torch.cat(pe_list, dim=-1)

    if normalize:
        pe = (pe - pe.mean()) / (pe.std() * 10)

    return pe

def build_2d_sincos_pe(L:int, V:int, D:int, device):

    assert D % 4 == 0, "d_model must be divisible by 4 (for 2D PE)"
    dim = D // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device) / dim))

    pos_t = torch.arange(L, device=device)[:, None]
    sin_t = torch.sin(pos_t * inv_freq)
    cos_t = torch.cos(pos_t * inv_freq)
    pe_t  = torch.cat([sin_t, cos_t], dim=-1)

    pos_c = torch.arange(V, device=device)[:, None]
    sin_c = torch.sin(pos_c * inv_freq)
    cos_c = torch.cos(pos_c * inv_freq)
    pe_c  = torch.cat([sin_c, cos_c], dim=-1)

    pe_2d = torch.zeros(L, V, D, device=device)
    pe_2d[..., :dim]  = pe_t[:, None, :]
    pe_2d[..., dim:]  = pe_c[None, :, :]
    return pe_2d
