"""
Unit tests for BasicBlock and conditional gating.
"""

import torch
import unittest
from mmseg_custom.models.backbones.basic_block import (
    ConditionalGatingBlock,
    ConditionalGatingBlockNoGating,
)


class TestConditionalGatingBlock(unittest.TestCase):
    """Test ConditionalGatingBlock."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.in_channels = 96
        self.out_channels = 192
        self.height, self.width = 32, 32
    
    def test_forward_with_condition(self):
        """Test forward pass with weather condition."""
        block = ConditionalGatingBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stage_idx=1,
            use_gating=True,
            use_condition=True,
        ).to(self.device)
        
        x = torch.randn(
            self.batch_size, self.in_channels, self.height, self.width,
            device=self.device
        )
        weather_label = torch.randint(0, 5, (self.batch_size,), device=self.device)
        
        output = block(x, weather_label=None)  # condition embedding inside
        
        # Condition embedding should be generated internally
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.height, self.width))
    
    def test_forward_without_condition(self):
        """Test forward pass without condition input."""
        block = ConditionalGatingBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            use_gating=False,
            use_condition=False,
        ).to(self.device)
        
        x = torch.randn(
            self.batch_size, self.in_channels, self.height, self.width,
            device=self.device
        )
        output = block(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.height, self.width))
    
    def test_gating_shape_mismatch(self):
        """Test that gating handles shape mismatches gracefully."""
        block = ConditionalGatingBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            use_gating=True,
            use_condition=True,
        ).to(self.device)
        
        x = torch.randn(
            self.batch_size, self.in_channels, self.height, self.width,
            device=self.device
        )
        
        # Pass wrong shape condition (should raise error)
        wrong_condition = torch.randn(self.batch_size, self.out_channels + 1, device=self.device)
        
        with self.assertRaises(ValueError):
            _ = block._apply_gating(x, wrong_condition)
    
    def test_no_gating_block(self):
        """Test ConditionalGatingBlockNoGating variant."""
        block = ConditionalGatingBlockNoGating(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        ).to(self.device)
        
        x = torch.randn(
            self.batch_size, self.in_channels, self.height, self.width,
            device=self.device
        )
        output = block(x, condition_embedding=None)
        
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.height, self.width))


if __name__ == '__main__':
    unittest.main()