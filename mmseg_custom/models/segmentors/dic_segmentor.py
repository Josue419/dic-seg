"""
Complete DiC Segmentation Model integrating Encoder + Decoder + Head.

This is the main model interface for MMSegmentation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from mmengine.model import BaseModel
from mmseg.registry import MODELS
from mmseg.models import build_backbone, build_head, build_loss

from ..backbones.dic_encoder import DicEncoder
from ..backbones.dic_decoder import DicDecoder


@MODELS.register_module()
class DicSegmentor(BaseModel):
    """Complete DiC Semantic Segmentation Model."""
    
    def __init__(
        self,
        arch: str = 'S',
        num_classes: int = 19,
        use_gating: bool = True,
        use_condition: bool = True,
        use_sparse_skip: bool = True,
        num_weather_classes: int = 5,
        with_auxiliary_head: bool = False,
        loss_decode: Optional[Dict] = None,
        auxiliary_head: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.arch = arch
        self.num_classes = num_classes
        self.use_condition = use_condition
        self.use_gating = use_gating
        self.use_sparse_skip = use_sparse_skip
        self.with_auxiliary_head = with_auxiliary_head
        
        # Encoder
        self.encoder = DicEncoder(
            arch=arch,
            use_gating=use_gating,
            use_condition=use_condition,
            use_sparse_skip=use_sparse_skip,
            in_channels=3,
            num_weather_classes=num_weather_classes,
        )
        
        # Decoder
        self.decoder = DicDecoder(
            arch=arch,
            use_gating=use_gating,
            use_sparse_skip=use_sparse_skip,
        )
        
        # Segmentation head
        d0_out_channels = self.decoder.stage_channels[0]
        self.decode_head = nn.Sequential(
            nn.Conv2d(d0_out_channels, num_classes, kernel_size=1, bias=True),
        )
        
        # Optional: Auxiliary head
        if with_auxiliary_head and auxiliary_head is not None:
            self.auxiliary_head = build_head(auxiliary_head)
        else:
            self.auxiliary_head = None
        
        # Loss function
        if loss_decode is None:
            loss_decode = dict(type='CrossEntropyLoss', use_sigmoid=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List] = None,
        mode: str = 'tensor',
        **kwargs,
    ):
        """Forward method with input validation."""
        
        # ✅ 关键调试：验证输入类型
        if not isinstance(inputs, torch.Tensor):
            print(f"❌ ERROR: inputs should be torch.Tensor, got {type(inputs)}")
            if isinstance(inputs, list):
                print(f"  inputs is list with length: {len(inputs)}")
                print(f"  first element type: {type(inputs[0]) if inputs else 'empty'}")
            raise TypeError(f"Expected torch.Tensor, got {type(inputs)}")
        
        print(f"✅ DEBUG: inputs type: {type(inputs)}, shape: {inputs.shape}")
        
        if mode == 'tensor':
            return self.forward_tensor(inputs, data_samples)
        elif mode == 'predict':
            return self.forward_predict(inputs, data_samples)
        elif mode == 'loss':
            return self.forward_loss(inputs, data_samples)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward_tensor(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List] = None,
    ) -> torch.Tensor:
        """Forward pass returning segmentation logits."""
        weather_label = None
        
        # 提取天气标签
        if data_samples is not None and len(data_samples) > 0:
            batch_size = inputs.shape[0]
            weather_labels = []
            
            for i in range(batch_size):
                weather = 0  # 默认值
                
                try:
                    if i < len(data_samples):
                        sample = data_samples[i]
                        
                        # 多种方式提取天气标签
                        if hasattr(sample, 'metainfo') and sample.metainfo is not None:
                            weather = sample.metainfo.get('weather_label', 0)
                            print(f"✅ Found weather_label in metainfo: {weather}")
                        elif hasattr(sample, 'weather_label'):
                            weather = sample.weather_label
                            print(f"✅ Found weather_label as attribute: {weather}")
                        elif isinstance(sample, dict):
                            weather = sample.get('weather_label', 0)
                            print(f"✅ Found weather_label in dict: {weather}")
                        else:
                            print(f"⚠ No weather_label found for sample {i}, using default: 0")
                    
                    weather = int(weather) if weather is not None else 0
                    if not (0 <= weather <= 4):
                        weather = 0
                        
                except Exception as e:
                    print(f"⚠ Warning: Failed to extract weather label for sample {i}: {e}")
                    weather = 0
                
                weather_labels.append(weather)
            
            # 创建天气标签张量
            if weather_labels:
                weather_label = torch.tensor(
                    weather_labels, dtype=torch.long, device=inputs.device
                )
                print(f"✅ Weather labels tensor: {weather_label}")
        
        # 前向传播
        encoder_outputs = self.encoder(inputs, weather_label)
        decoder_output = self.decoder(encoder_outputs)
        logits = self.decode_head(decoder_output)
        
        return logits
    
    def forward_predict(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List] = None,
    ) -> List[Dict]:
        """Forward pass for prediction."""
        logits = self.forward_tensor(inputs, data_samples)
        pred_label = logits.argmax(dim=1)
        
        predictions = []
        for i in range(pred_label.shape[0]):
            predictions.append({'segmentation': pred_label[i]})
        
        return predictions
    
    def forward_loss(
        self,
        inputs: torch.Tensor,
        data_samples: List,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training with loss."""
        logits = self.forward_tensor(inputs, data_samples)
        
        gt_labels = torch.stack([
            sample.gt_sem_seg.data.squeeze()
            for sample in data_samples
        ])

        loss = self.criterion(logits, gt_labels)
        
        losses = {'loss_seg': loss}
        
        if self.auxiliary_head is not None:
            pass
        
        return losses