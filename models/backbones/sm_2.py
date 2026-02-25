# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from functools import partial
import torch.nn.functional as F

from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init

from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw

# ---------------- Import Mamba ----------------
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    raise ImportError("Please install mamba_ssm (e.g., pip install mamba-ssm).")

# ---------------- DropPath (Safe) ----------------
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ---------------- Bidirectional Mamba Encoder Layer ----------------
class BidirectionalMambaEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims: int,
                 d_state: int = 16,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg=dict(type='LN'),
                 with_cp: bool = False):
        super().__init__()
        self.embed_dims = embed_dims
        self.with_cp = with_cp

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # Two independent Mamba instances (forward & backward)
        self.mamba_f = Mamba(d_model=embed_dims, d_state=d_state)
        self.mamba_b = Mamba(d_model=embed_dims, d_state=d_state)

        # Gating mechanism: split into ssm_out and z
        self.gate_proj = nn.Linear(embed_dims, 2 * embed_dims, bias=True)

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """
        Initialize only non-Mamba parameters.
        Mamba modules have their own specialized initialization (e.g., dt_proj bias).
        """
        for name, m in self.named_modules():
            # Skip Mamba's internal parameters to preserve dt_proj bias, etc.
            if 'mamba_f' in name or 'mamba_b' in name:
                continue
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)

    def forward_impl(self, x):
        # x: (B, N, C)
        identity = x
        normed = self.norm(x)

        # Forward Mamba
        y_f = self.mamba_f(normed)  # (B, N, C)

        # Backward Mamba: reverse sequence -> process -> reverse back
        y_b = self.mamba_b(normed.flip(1)).flip(1)  # (B, N, C)

        # Aggregate bidirectional outputs
        y = y_f + y_b  # (B, N, C)

        # Gating
        gate_in = self.gate_proj(y)  # (B, N, 2C)
        ssm_out, z = gate_in.split(self.embed_dims, dim=-1)
        out_core = ssm_out * F.silu(z)

        # Dropout + residual
        out_core = self.dropout(out_core)
        out = identity + self.drop_path(out_core)
        return out

    def forward(self, x, hw_shape):
        """
        Support gradient checkpointing if enabled and x.requires_grad.
        Note: Use with caution in WSL2 — may cause CUDA hang if mamba_ssm not fully compatible.
        """
        if self.with_cp and x.requires_grad:
            # Wrap to ensure only one positional arg for checkpoint
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module.forward_impl(*inputs)
                return custom_forward
            return cp.checkpoint(create_custom_forward(self), x, preserve_rng_state=True)
        else:
            return self.forward_impl(x)


# ---------------- MixVisionMamba Backbone ----------------
@MODELS.register_module()
class MixVisionMamba(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 d_state=16,
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('pretrained is deprecated, use init_cfg instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]

        cur = 0
        self.layers = ModuleList()
        in_ch = in_channels
        for i, nl in enumerate(num_layers):
            stage_dim = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_ch,
                embed_dims=stage_dim,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg
            )
            blocks = ModuleList([
                BidirectionalMambaEncoderLayer(
                    embed_dims=stage_dim,
                    d_state=d_state,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur + j],
                    norm_cfg=norm_cfg,
                    with_cp=with_cp
                ) for j in range(nl)
            ])
            stage_norm = build_norm_layer(norm_cfg, stage_dim)[1]
            self.layers.append(ModuleList([patch_embed, blocks, stage_norm]))
            cur += nl
            in_ch = stage_dim

    def init_weights(self):
        if self.init_cfg is None:
            for name, m in self.named_modules():
                # Skip Mamba parameters (they have custom init)
                if 'mamba_f' in name or 'mamba_b' in name:
                    continue
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.layers):
            patch_embed, blocks, norm = stage
            x, hw_shape = patch_embed(x)  # (B, N, C)
            for blk in blocks:
                x = blk(x, hw_shape)      # (B, N, C)
            x = norm(x)
            x = nlc_to_nchw(x, hw_shape)  # (B, C, H, W)
            if i in self.out_indices:
                outs.append(x)
        return outs