"""
Unit tests for complete DiC Segmentor model.
"""

import torch
import unittest
from mmseg_custom.models.segmentors.dic_segmentor import DicSegmentor


class TestDicSegmentor(unittest.TestCase):
    """Test complete DiC Segmentor."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.image_size = 512
        self.num_classes = 19
    
    def test_forward_tensor_mode(self):
        """Test forward pass in 'tensor' mode."""
        model = DicSegmentor(
            arch='S',
            num_classes=self.num_classes,
            use_gating=True,
            use_condition=True,
        ).to(self.device)
        
        x = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size,
            device=self.device
        )
        
        output = model.forward_tensor(x)
        
        self.assertEqual(
            output.shape,
            (self.batch_size, self.num_classes, self.image_size, self.image_size)
        )
    
    def test_forward_predict_mode(self):
        """Test forward pass in 'predict' mode."""
        model = DicSegmentor(arch='S', num_classes=self.num_classes).to(self.device)
        
        x = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size,
            device=self.device
        )
        
        predictions = model.forward_predict(x)
        
        self.assertEqual(len(predictions), self.batch_size)
        self.assertIn('segmentation', predictions[0])
        self.assertEqual(
            predictions[0]['segmentation'].shape,
            (self.image_size, self.image_size)
        )
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through model."""
        model = DicSegmentor(arch='S', num_classes=self.num_classes).to(self.device)
        
        x = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size,
            device=self.device, requires_grad=True
        )
        
        output = model.forward_tensor(x)
        loss = output.mean()
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None")


if __name__ == '__main__':
    unittest.main()