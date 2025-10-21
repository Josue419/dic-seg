"""
BasicBlock with Conditional Gating for DiC-Segmentation.

This module implements a fundamental building block that includes:
- Standard Conv3x3 + GroupNorm + GELU layers (no residual)
- Weather condition gating via Linear scale/shift
- Stage-specific embedding injection
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConditionalGatingBlock(nn.Module):
    """
    Basic Conv Block with Conditional Gating.
    
    Architecture:
        Conv3x3 → GroupNorm → GELU → 
        Conv3x3 → GroupNorm → GELU → 
        Conditional Gate (scale/shift)
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stage_idx (int): Stage index (0-4), used for dimension mapping in ablation.
        use_gating (bool): Whether to apply conditional gating. Default: True.
        use_condition (bool): Whether to accept condition input. Default: True.
        num_groups (int): Number of groups for GroupNorm. Default: 16.
        kernel_size (int): Kernel size of convolutions. Default: 3.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_idx: int = 0,
        use_gating: bool = True,
        use_condition: bool = True,
        num_groups: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage_idx = stage_idx
        self.use_gating = use_gating
        self.use_condition = use_condition
        
        padding = (kernel_size - 1) // 2
        
        # First conv block: Conv3x3 → GroupNorm → GELU
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act1 = nn.GELU()
        
        # Second conv block: Conv3x3 → GroupNorm → GELU
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act2 = nn.GELU()
        
        # Conditional gating (optional)
        if self.use_gating and self.use_condition:
            # Gate: Linear mapping from condition embedding → scale/shift
            # Expects condition embedding dim = out_channels
            self.gate_fc = nn.Linear(out_channels, 2 * out_channels)
            self.gate_fc.weight.data.normal_(0, 0.01)
            self.gate_fc.bias.data.zero_()
    
    def forward(
        self,
        x: torch.Tensor,
        condition_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature map, shape [B, C_in, H, W].
            condition_embedding (torch.Tensor, optional): 
                Condition embedding, shape [B, out_channels].
                If None and use_condition=True, treated as zero embedding.
        
        Returns:
            torch.Tensor: Output feature map, shape [B, C_out, H, W].
        """
        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        # Apply conditional gating if enabled
        if self.use_gating and self.use_condition:
            x = self._apply_gating(x, condition_embedding)
        
        return x
    
    def _apply_gating(
        self,
        x: torch.Tensor,
        condition_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply conditional gating: x = x * (1 + scale) + shift.
        
        Args:
            x (torch.Tensor): Feature map, shape [B, C, H, W].
            condition_embedding (torch.Tensor, optional): Shape [B, C].
        
        Returns:
            torch.Tensor: Gated feature map, shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        
        # Handle missing condition embedding
        if condition_embedding is None:
            # Zero embedding (neutral gating: scale=0, shift=0)
            condition_embedding = torch.zeros(B, C, dtype=x.dtype, device=x.device)
        
        # Ensure correct shape
        if condition_embedding.shape != (B, C):
            raise ValueError(
                f"Condition embedding shape {condition_embedding.shape} "
                f"does not match expected [B={B}, C={C}]"
            )
        
        # Linear mapping: [B, C] → [B, 2C]
        gate_params = self.gate_fc(condition_embedding)  # [B, 2C]
        scale, shift = gate_params.chunk(2, dim=1)  # each [B, C]
        
        # Reshape for broadcast: [B, C] → [B, C, 1, 1]
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # Apply gating: x = x * (1 + scale) + shift
        x = x * (1 + scale) + shift
        
        return x


class ConditionalGatingBlockNoGating(nn.Module):
    """
    BasicBlock without conditional gating (for ablation study: DiC w/o Conditional Gating).
    Only Conv3x3 + GroupNorm + GELU, but accepts condition_embedding parameter for API consistency.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_idx: int = 0,
        num_groups: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act2 = nn.GELU()
    
    def forward(
        self,
        x: torch.Tensor,
        condition_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass without gating. Condition ignored."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        return x