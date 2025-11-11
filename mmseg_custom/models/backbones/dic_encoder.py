"""
DiC Encoder with 5 stages (E0-E4) and Stage-Specific Weather Conditioning.

Each stage has:
- Multiple ConditionalGatingBlocks
- Weather embedding table (stage-specific dimension)
- Downsampling (AvgPool2D for E1-E4)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
from .basic_block import ConditionalGatingBlock, ConditionalGatingBlockNoGating


# Encoder configuration for DiC-S, DiC-B, DiC-XL
ENCODER_CONFIGS = {
    'S': {  # DiC-Small
        'num_blocks': [6, 6, 5, 6, 6],          # E0-E4
        'channels': [96, 192, 384, 192, 96],   # out_channels for each stage
    },
    'B': {  # DiC-Base
        'num_blocks': [8, 8, 6, 8, 8],
        'channels': [128, 256, 512, 1024, 1024],
    },
    'XL': {  # DiC-XLarge
        'num_blocks': [10, 10, 8, 10, 10],
        'channels': [160, 320, 640, 1280, 1280],
    },
}


class DicEncoderStage(nn.Module):
    """
    Single encoder stage with multiple blocks and weather conditioning.
    
    Args:
        stage_idx (int): Stage index (0-4).
        in_channels (int): Input channel count.
        out_channels (int): Output channel count.
        num_blocks (int): Number of ConditionalGatingBlocks in this stage.
        use_gating (bool): Enable conditional gating.
        use_condition (bool): Enable condition input.
        num_weather_classes (int): Number of weather classes (for embedding).
    """
    
    def __init__(
        self,
        stage_idx: int,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        use_gating: bool = True,
        use_condition: bool = True,
        num_weather_classes: int = 5,
    ):
        super().__init__()
        self.stage_idx = stage_idx
        self.out_channels = out_channels
        self.use_condition = use_condition
        
        # Stage-specific weather embedding table
        # Dimension = out_channels (matched to gating input)
        if use_condition:
            self.weather_embedding = nn.Embedding(
                num_embeddings=num_weather_classes,
                embedding_dim=out_channels
            )
            # Initialize embeddings
            nn.init.normal_(self.weather_embedding.weight, mean=0, std=0.01)
        
        # Blocks: first block may have channel change
        blocks = []
        for block_idx in range(num_blocks):
            block_in_ch = in_channels if block_idx == 0 else out_channels
            
            if use_gating:
                block = ConditionalGatingBlock(
                    in_channels=block_in_ch,
                    out_channels=out_channels,
                    stage_idx=stage_idx,
                    use_gating=True,
                    use_condition=use_condition,
                )
            else:
                block = ConditionalGatingBlockNoGating(
                    in_channels=block_in_ch,
                    out_channels=out_channels,
                    stage_idx=stage_idx,
                )
            blocks.append(block)
        
        self.blocks = nn.ModuleList(blocks)
        
        # Downsampling (except E0)
        if stage_idx > 0:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = None
    
    def forward(
        self,
        x: torch.Tensor,
        weather_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature map, shape [B, C_in, H, W].
            weather_label (torch.Tensor, optional): Weather class indices, shape [B].
                If None and use_condition=True, defaults to 0 (clear).
        
        Returns:
            torch.Tensor: Output feature map, shape [B, C_out, H/2, W/2] (if downsample).
        """
        # Get condition embedding
        condition_embedding = None
        if self.use_condition:
            if weather_label is None:
                # Default to clear (class 0)
                weather_label = torch.zeros(
                    x.shape[0], dtype=torch.long, device=x.device
                )
            condition_embedding = self.weather_embedding(weather_label)  # [B, C_out]
        
        # Pass through blocks with shared condition embedding
        for block in self.blocks:
            x = block(x, condition_embedding)
        
        # Apply downsampling
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x


class DicEncoder(nn.Module):
    """
    Complete DiC Encoder with 5 stages (E0-E4).
    
    Args:
        arch (str): Architecture size: 'S' (Small), 'B' (Base), 'XL' (XLarge).
        use_gating (bool): Enable conditional gating across all stages. Default: True.
        use_condition (bool): Enable weather condition input. Default: True.
        use_sparse_skip (bool): For consistency API (actual skip in decoder). Default: True.
        in_channels (int): Input image channels (RGB=3).
        num_weather_classes (int): Number of weather classes. Default: 5.
    """
    
    def __init__(
        self,
        arch: str = 'S',
        use_gating: bool = True,
        use_condition: bool = True,
        use_sparse_skip: bool = True,
        in_channels: int = 3,
        num_weather_classes: int = 5,
    ):
        super().__init__()
        
        if arch not in ENCODER_CONFIGS:
            raise ValueError(f"Architecture {arch} not in {ENCODER_CONFIGS.keys()}")
        
        config = ENCODER_CONFIGS[arch]
        self.arch = arch
        self.use_condition = use_condition
        self.use_sparse_skip = use_sparse_skip
        
        # Build 5 stages
        stages = []
        prev_channels = in_channels
        
        for stage_idx, (num_blocks, out_ch) in enumerate(
            zip(config['num_blocks'], config['channels'])
        ):
            stage = DicEncoderStage(
                stage_idx=stage_idx,
                in_channels=prev_channels,
                out_channels=out_ch,
                num_blocks=num_blocks,
                use_gating=use_gating,
                use_condition=use_condition,
                num_weather_classes=num_weather_classes,
            )
            stages.append(stage)
            prev_channels = out_ch
        
        self.stages = nn.ModuleList(stages)
        
        # Store for reference
        self.stage_channels = config['channels']
    
    def forward(
        self,
        x: torch.Tensor,
        weather_label: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input image, shape [B, 3, H, W].
            weather_label (torch.Tensor, optional): Weather class, shape [B].
        
        Returns:
            List[torch.Tensor]: List of 5 stage outputs (skip connections for decoder).
                - Each output has shape [B, C_stage, H/2^stage, W/2^stage]
                - E0: [B, 96, H, W]
                - E1: [B, 192, H/2, W/2]
                - E2: [B, 384, H/4, W/4]
                - E3: [B, 768, H/8, W/8]
                - E4: [B, 768, H/16, W/16]
        """
        skip_features = []
        
        for stage in self.stages:
            x = stage(x, weather_label)
            skip_features.append(x)
        
        return skip_features