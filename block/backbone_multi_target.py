from typing import Optional

import torch
from torch import nn, Tensor

from block.attention import MultiheadAttention
from block.utils import Transpose, get_activation_fn
from block.Positionemb import build_2d_sincos_pe

class mmJEPA(nn.Module):

    def __init__(self, c_in: int, target_dim: int, patch_len: int, num_patch: int,
                 n_layers: int = 3, d_model=128, n_heads=16, d_ff: int = 256,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 head_dropout=0,
                 head_type="prediction"):

        super().__init__()

        self.backbone = context_encoder(c_in, num_patch=num_patch, patch_len=patch_len,
                                        n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                                        d_ff=d_ff)

        self.n_vars = c_in
        self.head_type = head_type

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len,
                                     head_dropout)

        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)

    def forward(self, z):

        z = self.backbone(z)
        z = self.head(z)

        return z

class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):

        x = x[:, :, :, -1]
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)
        return y

class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):

        x = x.transpose(2, 3)
        x = self.linear(self.dropout(x))
        x = x.permute(0, 2, 1, 3)
        return x

class Projection(nn.Module):

    def __init__(self, patch_len:int, d_model:int,
                 option:str='linear'):
        super().__init__()
        if option == 'dwconv':
            self.proj = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=3, padding=1, groups=1, bias=False),
                nn.Flatten(1,-1),
                nn.Linear(patch_len, d_model, bias=False))
        else:
            self.proj = nn.Linear(patch_len, d_model, bias=False)

    def forward(self, x):
        B,L,V,P = x.shape
        x = self.proj(x.reshape(B*L*V, -1))
        return x.view(B, L, V, -1)

class context_encoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len,
                 n_layers=3, d_model=192, n_heads=8,
                 d_ff=512, dp=0., device=None):
        super().__init__()
        self.n_vars, self.d_model = c_in, d_model
        self.L = num_patch
        self.V = c_in

        self.W_P = Projection(patch_len, d_model, option='linear')

        pe_2d = build_2d_sincos_pe(self.L, self.V, d_model, device)
        self.register_buffer('W_pos', pe_2d)

        self.space_proj = nn.Linear(3, d_model, bias=False)

        self.drop = nn.Dropout(0.)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder = TSTEncoder(d_model, n_heads,
                                  d_ff=d_ff, res_attention=True,norm='LayerNorm',
                                  n_layers=n_layers, pre_norm=True, dropout=dp)

    def forward(self, x, *, position=None,
                obs_m):

        B, L, V, _ = x.shape
        D = self.d_model
        dev = x.device

        x = self.W_P(x)

        x = self.drop(x + self.W_pos)
        if position is not None:
            x = x + self.space_proj(position)[:,None,:,:]

        if obs_m is not None:
            x_obs = x[obs_m].view(B, -1, D)
            s_x = self.encoder(x_obs)
            mask4d = obs_m.unsqueeze(-1).expand(-1, -1, -1, D)
            z_pad = (self.mask_token.data + self.W_pos.to(torch.float32)).expand_as(x).clone()
            if position is not None:
                z_pad += self.space_proj(position)[:, None, :, :]
            z_pad.masked_scatter_(mask4d, s_x.reshape(-1))

            return s_x, z_pad

        else:
            x = x.view(x.size(0), -1, D)
            s_x = self.encoder(x)
            s_x = s_x.view(B,L,V,D)
            return s_x

class attn_MultiMaskPredictor(nn.Module):
    def __init__(self, d_model=192, n_heads=4, d_ff=192, n_layers=1, dropout=0.):
        super().__init__()

        self.query_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query_tok, std=0.02)

        self.head = TSTEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            pre_norm=True,
            dropout=dropout
        )

        self.hr_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, z_pad, masks):

        B, L, V, D = z_pad.shape
        ctx = z_pad.view(B, -1, D)
        out_seq = self.head(ctx)
        out_seq = out_seq.view(B, L, V, D)

        hr_feat = out_seq.mean(dim=(1, 2))
        hr_pred = self.hr_predictor(hr_feat)

        outs = []
        for m in masks:

            pred = out_seq[m].view(B, -1, D)
            outs.append(pred)

        outs.append(hr_pred)
        return tuple(outs)

class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False, **kwargs):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model
        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     activation=activation, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn) for i in
                                     range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor):

        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output

class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0., dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout,
                                            res_attention=res_attention)

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None):

        if self.pre_norm:
            src = self.norm_attn(src)

        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn

        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)

        src2 = self.ff(src)

        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src
