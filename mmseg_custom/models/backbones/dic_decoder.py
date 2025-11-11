"""
DiC Decoder with 5 stages (D4-D0) symmetric to encoder.

Key design:
- Upsampling by ConvTranspose2d (k=2, s=2)
- Stage-level skip connections (concat encoder output)
- No weather conditioning in decoder (only in encoder)

修改说明：
- 修复 Decoder 维度计算以适配对称 Encoder 配置 [96, 192, 384, 192, 96]
- 确保 skip 连接维度匹配（D3 skip from E3=192，不是 768）
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .basic_block import ConditionalGatingBlock, ConditionalGatingBlockNoGating


class DicDecoderStage(nn.Module):
    """
    Single decoder stage with skip connection and upsampling.
    
    Flow:
        Input → Upsample → Concat with skip → Conv blocks → Output
    
    Args:
        stage_idx (int): Decoder stage index (4-0).
        in_channels (int): Input channel count (from previous decoder stage or bottleneck).
        skip_channels (int): Skip connection channel count (from encoder).
        out_channels (int): Output channel count.
        num_blocks (int): Number of ConditionalGatingBlocks.
        use_gating (bool): Enable conditional gating (ablation study).
        is_bottleneck (bool): Whether this is bottleneck (D4, no upsample/skip).
    """
    
    def __init__(
        self,
        stage_idx: int,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_blocks: int,
        use_gating: bool = True,
        is_bottleneck: bool = False,
    ):
        super().__init__()
        self.stage_idx = stage_idx
        self.out_channels = out_channels
        self.is_bottleneck = is_bottleneck
        
        # Upsample (except bottleneck)
        if not is_bottleneck:
            self.upsample = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=2,
                stride=2,
                bias=False
            )
            concat_channels = in_channels + skip_channels
        else:
            self.upsample = None
            concat_channels = in_channels
        
        # Blocks: first block may have channel change from concat
        blocks = []
        for block_idx in range(num_blocks):
            block_in_ch = concat_channels if block_idx == 0 else out_channels
            
            if use_gating:
                block = ConditionalGatingBlock(
                    in_channels=block_in_ch,
                    out_channels=out_channels,
                    stage_idx=stage_idx,
                    use_gating=False,  # No condition in decoder
                    use_condition=False,
                )
            else:
                block = ConditionalGatingBlockNoGating(
                    in_channels=block_in_ch,
                    out_channels=out_channels,
                    stage_idx=stage_idx,
                )
            blocks.append(block)
        
        self.blocks = nn.ModuleList(blocks)
    
    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input from bottleneck or previous decoder stage.
            skip (torch.Tensor, optional): Skip connection from encoder.
        
        Returns:
            torch.Tensor: Output feature map.
        """
        # Upsample (if not bottleneck)
        if self.upsample is not None:
            x = self.upsample(x)
        
        # Concatenate skip connection (if provided)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x, condition_embedding=None)
        
        return x


class DicDecoder(nn.Module):
    """
    Complete DiC Decoder with 5 stages (D4-D0), symmetric to encoder.
    
    Structure:
        D4 (bottleneck) → D3 (upsample + skip E3) → D2 → D1 → D0
    
    Key fix for symmetric encoder:
    - When encoder channels are [96, 192, 384, 192, 96]
    - D4: in=96, out=96 (bottleneck)
    - D3: in=96, skip=192 (from E3), out=192
    - D2: in=192, skip=384 (from E2), out=384
    - D1: in=384, skip=192 (from E1), out=192
    - D0: in=192, skip=96 (from E0), out=96
    
    Args:
        arch (str): Architecture size: 'S', 'B', 'XL' (must match encoder).
        use_gating (bool): Enable conditional gating (for encoder-like API).
        use_sparse_skip (bool): Use sparse (stage-level) skip. Default: True.
    """
    
    def __init__(
        self,
        arch: str = 'S',
        use_gating: bool = True,
        use_sparse_skip: bool = True,
    ):
        super().__init__()
        
        from .dic_encoder import ENCODER_CONFIGS
        
        if arch not in ENCODER_CONFIGS:
            raise ValueError(f"Architecture {arch} not in {ENCODER_CONFIGS.keys()}")
        
        config = ENCODER_CONFIGS[arch]
        self.arch = arch
        self.use_sparse_skip = use_sparse_skip
        
        # Decoder mirrors encoder configuration
        num_blocks_list = config['num_blocks']  # [6, 6, 5, 6, 6] for E0-E4
        channels_list = config['channels']      # [96, 192, 384, 192, 96] for symmetric S
        
        # Build decoder stages: D4, D3, D2, D1, D0
        # D4 is bottleneck (no upsample, no skip)
        # D3-D0 have upsample and skip
        
        stages = []
        
        # D4: bottleneck stage
        # Input from encoder E4, no upsampling, no skip connection
        d4 = DicDecoderStage(
            stage_idx=4,
            in_channels=channels_list[4],    # E4 output channels
            skip_channels=0,                 # No skip for bottleneck
            out_channels=channels_list[4],   # D4 output = E4 channels
            num_blocks=num_blocks_list[4],   # Same block count as E4
            use_gating=use_gating,
            is_bottleneck=True,
        )
        stages.append(d4)
        
        # D3-D0: symmetric to E3-E0
        # 🔑 Critical fix: Iterate in reverse order (3, 2, 1, 0)
        for stage_idx in range(3, -1, -1):
            # D{stage_idx} receives input from D{stage_idx+1} output
            # And skip connection from E{stage_idx} output
            d_stage = DicDecoderStage(
                stage_idx=stage_idx,
                # Input: comes from previous decoder stage output (which has channels_list[stage_idx+1])
                in_channels=channels_list[stage_idx + 1],
                # Skip: comes from corresponding encoder stage (has channels_list[stage_idx])
                skip_channels=channels_list[stage_idx],
                # Output: matches encoder stage dimension
                out_channels=channels_list[stage_idx],
                # Block count: same as encoder stage
                num_blocks=num_blocks_list[stage_idx],
                use_gating=use_gating,
                is_bottleneck=False,
            )
            stages.append(d_stage)
        
        self.stages = nn.ModuleList(stages)
        self.stage_channels = channels_list
    
    def forward(
        self,
        encoder_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            encoder_outputs (List[torch.Tensor]): Encoder stage outputs (E0-E4).
                For DiC-S with symmetric config [96, 192, 384, 192, 96]:
                - encoder_outputs[0]: E0, shape [B, 96, H, W]
                - encoder_outputs[1]: E1, shape [B, 192, H/2, W/2]
                - encoder_outputs[2]: E2, shape [B, 384, H/4, W/4]
                - encoder_outputs[3]: E3, shape [B, 192, H/8, W/8]
                - encoder_outputs[4]: E4, shape [B, 96, H/16, W/16]
        
        Returns:
            torch.Tensor: Final decoder output, shape [B, 96, H, W].
        
        Flow:
            D4 (bottleneck):   E4 [B, 96, H/16, W/16] → [B, 96, H/16, W/16]
            ↓
            D3 (upsample+skip E3): [B, 96, H/16, W/16] 
                                   ↓ (upsample) → [B, 96, H/8, W/8]
                                   ↓ (concat E3) → [B, 96+192, H/8, W/8]
                                   ↓ (blocks) → [B, 192, H/8, W/8]
            ↓
            D2 (upsample+skip E2): [B, 192, H/8, W/8]
                                   ↓ (upsample) → [B, 192, H/4, W/4]
                                   ↓ (concat E2) → [B, 192+384, H/4, W/4]
                                   ↓ (blocks) → [B, 384, H/4, W/4]
            ↓
            D1 (upsample+skip E1): [B, 384, H/4, W/4]
                                   ↓ (upsample) → [B, 384, H/2, W/2]
                                   ↓ (concat E1) → [B, 384+192, H/2, W/2]
                                   ↓ (blocks) → [B, 192, H/2, W/2]
            ↓
            D0 (upsample+skip E0): [B, 192, H/2, W/2]
                                   ↓ (upsample) → [B, 192, H, W]
                                   ↓ (concat E0) → [B, 192+96, H, W]
                                   ↓ (blocks) → [B, 96, H, W]
        """
        # D4: bottleneck (no skip needed)
        x = self.stages[0](encoder_outputs[4], skip=None)
        
        # D3-D0: with skip connections
        # stages[0] is D4 (index 0)
        # stages[1] is D3 (index 1)
        # stages[2] is D2 (index 2)
        # stages[3] is D1 (index 3)
        # stages[4] is D0 (index 4)
        for stage_idx in range(3, -1, -1):
            # Get skip connection from corresponding encoder stage
            skip = encoder_outputs[stage_idx]
            
            # Get decoder stage (offset by 1 because stages[0] is D4)
            decoder_stage = self.stages[4 - stage_idx]
            
            # Forward through decoder stage
            x = decoder_stage(x, skip=skip)
        
        return x