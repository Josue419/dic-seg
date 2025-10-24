"""
Memory leak detection tests.

⚠️  ONLY FOR LOCAL DEBUGGING! Disable for cloud training.

Run with:
    python -m pytest tests/test_memory_leaks.py -v -s
"""

import torch
import unittest
from mmseg_custom.models.backbones.dic_encoder import DicEncoder
from mmseg_custom.models.backbones.dic_decoder import DicDecoder
from mmseg_custom.models.segmentors.dic_segmentor import DicSegmentor
from memory_check import batch_forward_memory_check, GPUMemoryTracker


class TestMemoryLeaks(unittest.TestCase):
    """Test for memory leaks in model components."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            self.skipTest("CUDA not available, skipping memory tests")
    
    def test_encoder_memory_leak(self):
        """Test DiC Encoder for memory leaks."""
        print("\n" + "="*70)
        print("Testing: DiC Encoder Memory Leak")
        print("="*70)
        
        encoder = DicEncoder(arch='S', use_condition=True).to(self.device)
        x = torch.randn(1, 3, 256, 256, device=self.device)
        
        stats, has_leak = batch_forward_memory_check(
            encoder,
            x,
            num_iterations=10,
            backward=False,
        )
        
        print(f"\nResult: {'❌ LEAK DETECTED' if has_leak else '✅ NO LEAK'}")
        print(f"Total growth: {stats['total_growth_mb']:.2f} MB")
        
        # Accept up to 200 MB growth (some growth is normal due to caching)
        self.assertLess(stats['total_growth_mb'], 200.0,
                        f"Memory growth too large: {stats['total_growth_mb']:.2f} MB")
    
    def test_decoder_memory_leak(self):
        """Test DiC Decoder for memory leaks."""
        print("\n" + "="*70)
        print("Testing: DiC Decoder Memory Leak")
        print("="*70)
        
        encoder = DicEncoder(arch='S').to(self.device)
        decoder = DicDecoder(arch='S').to(self.device)
        
        x = torch.randn(1, 3, 256, 256, device=self.device)
        encoder_outputs = encoder(x)
        
        # Create a dummy model that encapsulates decoder
        class DecoderWrapper(torch.nn.Module):
            def __init__(self, enc, dec):
                super().__init__()
                self.encoder = enc
                self.decoder = dec
            
            def forward(self, x):
                enc_out = self.encoder(x)
                return self.decoder(enc_out)
        
        wrapper = DecoderWrapper(encoder, decoder).to(self.device)
        
        stats, has_leak = batch_forward_memory_check(
            wrapper,
            x,
            num_iterations=10,
            backward=False,
        )
        
        print(f"\nResult: {'❌ LEAK DETECTED' if has_leak else '✅ NO LEAK'}")
        print(f"Total growth: {stats['total_growth_mb']:.2f} MB")
        
        self.assertLess(stats['total_growth_mb'], 250.0,
                        f"Memory growth too large: {stats['total_growth_mb']:.2f} MB")
    
    def test_full_model_memory_leak(self):
        """Test complete DicSegmentor for memory leaks."""
        print("\n" + "="*70)
        print("Testing: Full DicSegmentor Memory Leak")
        print("="*70)
        
        model = DicSegmentor(arch='S', num_classes=19).to(self.device)
        x = torch.randn(1, 3, 256, 256, device=self.device)
        
        stats, has_leak = batch_forward_memory_check(
            model,
            x,
            num_iterations=5,
            backward=True,  # Include backward for complete test
        )
        
        print(f"\nResult: {'❌ LEAK DETECTED' if has_leak else '✅ NO LEAK'}")
        print(f"Total growth: {stats['total_growth_mb']:.2f} MB")
        
        # Allow more growth for backward pass
        self.assertLess(stats['total_growth_mb'], 300.0,
                        f"Memory growth too large: {stats['total_growth_mb']:.2f} MB")
    
    def test_multiple_architectures_memory(self):
        """Test different architectures for memory consistency."""
        print("\n" + "="*70)
        print("Testing: Memory Consistency Across Architectures")
        print("="*70)
        
        for arch in ['S', 'B', 'XL']:
            print(f"\n--- Testing DiC-{arch} ---")
            
            model = DicSegmentor(arch=arch, num_classes=19).to(self.device)
            x = torch.randn(1, 3, 256, 256, device=self.device)
            
            tracker = GPUMemoryTracker(device=self.device)
            tracker.reset()
            
            # Run 3 iterations
            for i in range(3):
                output = model(x)
                stats = tracker.record_iteration(i)
                del output
                torch.cuda.empty_cache()
                
                print(f"  Iter {i}: {stats['current_mb']:.2f} MB (growth: {stats['growth_mb']:.2f} MB)")
            
            final_growth = tracker.iteration_memories[-1]['growth_mb']
            print(f"Final growth for DiC-{arch}: {final_growth:.2f} MB")


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency metrics."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            self.skipTest("CUDA not available")
    
    def test_activation_memory_ratio(self):
        """Check that activation memory is not excessive."""
        model = DicSegmentor(arch='S').to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        param_memory_mb = (total_params * 4) / 1e6  # FP32
        
        # Measure forward activation
        tracker = GPUMemoryTracker(device=self.device)
        tracker.reset()
        
        x = torch.randn(1, 3, 512, 512, device=self.device)
        with torch.no_grad():
            output = model(x)
        
        activation_memory_mb = tracker.get_memory_mb()
        
        print(f"\nParameter memory: {param_memory_mb:.2f} MB")
        print(f"Activation memory: {activation_memory_mb:.2f} MB")
        print(f"Ratio: {activation_memory_mb / param_memory_mb:.1f}x")
        
        # Activation should not be more than 10x parameter memory
        self.assertLess(activation_memory_mb / param_memory_mb, 10.0,
                       "Activation memory too large relative to parameters")


if __name__ == '__main__':
    unittest.main()