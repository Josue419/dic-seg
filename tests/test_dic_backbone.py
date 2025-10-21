"""
Unit tests for DiC Encoder and Decoder.
"""

import torch
import unittest
from mmseg_custom.models.backbones.dic_encoder import DicEncoder
from mmseg_custom.models.backbones.dic_decoder import DicDecoder


class TestDicEncoder(unittest.TestCase):
    """Test DicEncoder."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.image_size = 512
    
    def test_forward_shape_s(self):
        """Test encoder output shapes for DiC-S."""
        encoder = DicEncoder(arch='S', use_condition=True).to(self.device)
        
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device)
        weather_label = torch.randint(0, 5, (self.batch_size,), device=self.device)
        
        outputs = encoder(x, weather_label)
        
        # Should return 5 stage outputs
        self.assertEqual(len(outputs), 5)
        
        # Check shapes
        expected_shapes = [
            (self.batch_size, 96, 512, 512),    # E0
            (self.batch_size, 192, 256, 256),   # E1
            (self.batch_size, 384, 128, 128),   # E2
            (self.batch_size, 768, 64, 64),     # E3
            (self.batch_size, 768, 32, 32),     # E4
        ]
        
        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            self.assertEqual(output.shape, expected_shape, f"Stage E{i} shape mismatch")
    
    def test_forward_without_condition(self):
        """Test encoder without condition input."""
        encoder = DicEncoder(arch='S', use_condition=False).to(self.device)
        
        x = torch.randn(self.batch_size, 3, 512, 512, device=self.device)
        outputs = encoder(x, weather_label=None)
        
        self.assertEqual(len(outputs), 5)


class TestDicDecoder(unittest.TestCase):
    """Test DicDecoder."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
    
    def test_forward_shape_s(self):
        """Test decoder output shape for DiC-S."""
        encoder = DicEncoder(arch='S').to(self.device)
        decoder = DicDecoder(arch='S').to(self.device)
        
        x = torch.randn(self.batch_size, 3, 512, 512, device=self.device)
        encoder_outputs = encoder(x)
        decoder_output = decoder(encoder_outputs)
        
        # Decoder should output feature map with encoder E0 channel count
        self.assertEqual(decoder_output.shape, (self.batch_size, 96, 512, 512))


if __name__ == '__main__':
    unittest.main()