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
from mmseg.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    raise ImportError("Please ensure mamba_ssm is installed and available (hustvl/Vim submodule).")

class BidirectionalMambaEncoderLayer(BaseModule):
    """
    Enhanced Mamba Layer supporting:
    1. Bidirectional Mamba (Forward + Backward)
    2. Optional 3x3 Depthwise Convolution (Local Context)
    3. Configurable Scan Direction (Horizontal or Vertical)
    """
    def __init__(self,
                 embed_dims: int,
                 d_state: int = 16,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg=dict(type='LN'),
                 scan_type='horizontal',  # 'horizontal' or 'vertical'
                 with_dwconv=False,       # Whether to add 3x3 DWConv
                 with_cp: bool = False):
        super().__init__()
        self.embed_dims = embed_dims
        self.with_cp = with_cp
        self.scan_type = scan_type
        self.with_dwconv = with_dwconv

        # 1. Norm
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # 2. Optional Depthwise Convolution (for Stage 1 & 2)
        if self.with_dwconv:
            self.dwconv = nn.Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=3,
                padding=1,
                groups=embed_dims, # Depthwise
                bias=True
            )

        # 3. Two Mamba instances
        self.mamba_f = Mamba(d_model=embed_dims, d_state=d_state)
        self.mamba_b = Mamba(d_model=embed_dims, d_state=d_state)

        # 4. Gating projection
        self.gate_proj = nn.Linear(embed_dims, 2 * embed_dims, bias=True)

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward_impl(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        
        identity = x
        x_norm = self.norm(x)

        # --- Step 1: Optional Local Convolution ---
        if self.with_dwconv:
            # Reshape to NCHW for Conv2d
            x_spatial = nlc_to_nchw(x_norm, hw_shape)
            x_spatial = self.dwconv(x_spatial)
            x_norm = nchw_to_nlc(x_spatial)

        # --- Step 2: Handle Scan Direction ---
        if self.scan_type == 'vertical':
            # Reshape to (B, H, W, C) -> Transpose to (B, W, H, C) -> Flatten
            # This makes columns contiguous in memory
            x_mamba_in = x_norm.view(B, H, W, C).permute(0, 2, 1, 3).flatten(1, 2)
        else:
            # Horizontal (Standard)
            x_mamba_in = x_norm

        # --- Step 3: Bidirectional Mamba ---
        # Forward
        y_f = self.mamba_f(x_mamba_in)
        # Backward (reverse the sequence, run mamba, reverse back)
        y_b = self.mamba_b(x_mamba_in.flip(1)).flip(1)
        
        y = y_f + y_b

        # --- Step 4: Restore Shape if Vertical ---
        if self.scan_type == 'vertical':
            # Current y is (B, W*H, C) representing (B, W, H, C)
            # Transpose back: (B, W, H, C) -> (B, H, W, C) -> Flatten
            y = y.view(B, W, H, C).permute(0, 2, 1, 3).flatten(1, 2)

        # --- Step 5: Gating & Residual ---
        gate_in = self.gate_proj(y)
        ssm_out, z = gate_in.split(self.embed_dims, dim=-1)
        out_core = ssm_out * F.silu(z)
        out_core = self.dropout(out_core)
        
        out = identity + self.drop_path(out_core)
        return out

    def forward(self, x, hw_shape):
        if self.with_cp and x.requires_grad:
            return cp.checkpoint(self.forward_impl, x, hw_shape)
        else:
            return self.forward_impl(x, hw_shape)


class DropPath(nn.Module):
    """ Drop paths (Stochastic Depth) per sample. """
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


@MODELS.register_module()
class MixVisionMamba(BaseModule):
    """
    Custom Backbone with Scan-Direction mixing and DWConv injection.
    
    Structure:
    - Stage 1 (i=0): All Horizontal + DWConv (Strong local feat, standard row scan)
    - Stage 2 (i=1): All Horizontal + DWConv
    - Stage 3 (i=2): 
        - First 2 blocks: Horizontal (No DWConv)
        - Remaining blocks: Vertical (No DWConv) -> Capture electrical poles, tall objects
    - Stage 4 (i=3): All Vertical (No DWConv) -> Capture global vertical context
    """
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 out_indices=(0, 1, 2, 3),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 d_state=16,
                 init_cfg=None,
                 with_cp=False,
                 **kwargs): # Catch extra args
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.out_indices = out_indices
        self.with_cp = with_cp

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
        cur = 0
        
        self.layers = ModuleList()
        in_ch = in_channels

        for i, nl in enumerate(num_layers):
            stage_dim = embed_dims * num_heads[i]
            
            # Patch Embedding
            patch_embed = PatchEmbed(
                in_channels=in_ch,
                embed_dims=stage_dim,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg
            )

            # Blocks Construction
            blocks = ModuleList()
            for j in range(nl):
                # --- Configuration Logic based on User Request ---
                
                # 1. Determine DWConv: Stage 1 & 2 (indices 0, 1) get DWConv
                use_dwconv = True # if i < 2 else False
                
                # 2. Determine Scan Direction:
                # Stage 1, 2: Horizontal
                # Stage 3: First 2 Horizontal, Rest Vertical
                # Stage 4: Vertical
                scan_type = 'horizontal'
                if i == 2: # Stage 3
                    if j >= 2: # After first two blocks
                        scan_type = 'vertical'
                elif i == 3: # Stage 4
                    scan_type = 'vertical'
                
                blocks.append(
                    BidirectionalMambaEncoderLayer(
                        embed_dims=stage_dim,
                        d_state=d_state,
                        drop_rate=drop_rate,
                        drop_path_rate=dpr[cur + j],
                        norm_cfg=norm_cfg,
                        scan_type=scan_type,
                        with_dwconv=use_dwconv,
                        with_cp=with_cp
                    )
                )

            stage_norm = build_norm_layer(norm_cfg, stage_dim)[1]
            self.layers.append(ModuleList([patch_embed, blocks, stage_norm]))
            cur += nl
            in_ch = stage_dim

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
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
            x, hw_shape = patch_embed(x)
            for blk in blocks:
                x = blk(x, hw_shape)
            x = norm(x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)
        return outs