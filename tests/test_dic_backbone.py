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
        self.batch_size = 1  # ✅ 改为 1（显存减半）
        self.image_size = 256  # ✅ 改为 256（显存减 1/4）
    
    def tearDown(self):
        """清空显存缓存（重要！）"""
        torch.cuda.empty_cache()
    
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
            (self.batch_size, 96, self.image_size, self.image_size),              # E0: 256×256
            (self.batch_size, 192, self.image_size//2, self.image_size//2),       # E1: 128×128
            (self.batch_size, 384, self.image_size//4, self.image_size//4),       # E2: 64×64
            (self.batch_size, 768, self.image_size//8, self.image_size//8),       # E3: 32×32
            (self.batch_size, 768, self.image_size//16, self.image_size//16),     # E4: 16×16
        ]
        
        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            self.assertEqual(output.shape, expected_shape, f"Stage E{i} shape mismatch")
    
    def test_forward_without_weather_label(self):
        """Test encoder without weather_label (should default to 0)."""
        encoder = DicEncoder(arch='S', use_condition=True).to(self.device)
        
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device)
        
        # Pass None → should default to weather_label=0 (clear)
        outputs = encoder(x, weather_label=None)
        
        self.assertEqual(len(outputs), 5)
    
    def test_forward_without_condition(self):
        """Test encoder without condition input."""
        encoder = DicEncoder(arch='S', use_condition=False).to(self.device)
        
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device)
        outputs = encoder(x, weather_label=None)
        
        self.assertEqual(len(outputs), 5)
    
    def test_encoder_different_architectures(self):
        """Test encoder with different architecture sizes."""
        for arch in ['S', 'B', 'XL']:
            encoder = DicEncoder(arch=arch).to(self.device)
            
            x = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device)
            outputs = encoder(x)
            
            self.assertEqual(len(outputs), 5, f"Architecture {arch} should output 5 stages")
            
            # Clean up after each architecture test
            del encoder
            torch.cuda.empty_cache()


class TestDicDecoder(unittest.TestCase):
    """Test DicDecoder."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1  # ✅ 改为 1
        self.image_size = 256  # ✅ 改为 256
    
    def tearDown(self):
        """清空显存缓存"""
        torch.cuda.empty_cache()
    
    def test_forward_shape_s(self):
        """Test decoder output shape for DiC-S."""
        encoder = DicEncoder(arch='S').to(self.device)
        decoder = DicDecoder(arch='S').to(self.device)
        
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device)
        encoder_outputs = encoder(x)
        decoder_output = decoder(encoder_outputs)
        
        # Decoder should output feature map with encoder E0 channel count
        self.assertEqual(decoder_output.shape, (self.batch_size, 96, self.image_size, self.image_size))
    
    def test_decoder_shapes_all_stages(self):
        """Test decoder intermediate shapes."""
        encoder = DicEncoder(arch='S').to(self.device)
        decoder = DicDecoder(arch='S').to(self.device)
        
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device)
        encoder_outputs = encoder(x)
        
        # Verify encoder outputs before decoder
        expected_encoder_shapes = [
            (self.batch_size, 96, self.image_size, self.image_size),              # E0
            (self.batch_size, 192, self.image_size//2, self.image_size//2),       # E1
            (self.batch_size, 384, self.image_size//4, self.image_size//4),       # E2
            (self.batch_size, 768, self.image_size//8, self.image_size//8),       # E3
            (self.batch_size, 768, self.image_size//16, self.image_size//16),     # E4
        ]
        
        for i, (output, expected_shape) in enumerate(zip(encoder_outputs, expected_encoder_shapes)):
            self.assertEqual(output.shape, expected_shape, f"Encoder stage E{i} shape mismatch")
        
        # Decoder output
        decoder_output = decoder(encoder_outputs)
        self.assertEqual(decoder_output.shape, (self.batch_size, 96, self.image_size, self.image_size))
    
    def test_gradient_flow_encoder_decoder(self):
        """Test gradient flow through encoder and decoder."""
        encoder = DicEncoder(arch='S').to(self.device)
        decoder = DicDecoder(arch='S').to(self.device)
        
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device, requires_grad=True)
        
        encoder_outputs = encoder(x)
        decoder_output = decoder(encoder_outputs)
        loss = decoder_output.mean()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))


if __name__ == '__main__':
    unittest.main()