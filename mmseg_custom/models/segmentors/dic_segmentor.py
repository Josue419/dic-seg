"""
DiC Segmentation Model - 显存优化版本

关键优化：
1. 梯度检查点（Gradient Checkpointing）
2. 优化的天气标签处理
3. 内存友好的前向传播
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, List, Dict
from mmengine.model import BaseModel
from mmseg.registry import MODELS

from ..backbones.dic_encoder import DicEncoder
from ..backbones.dic_decoder import DicDecoder


@MODELS.register_module()
class DicSegmentor(BaseModel):
    """DiC Semantic Segmentation Model - 显存优化版"""
    
    def __init__(
        self,
        arch: str = 'S',
        num_classes: int = 19,
        use_gating: bool = True,
        use_condition: bool = True,
        use_sparse_skip: bool = True,
        num_weather_classes: int = 5,
        gradient_checkpointing: bool = False,
        memory_efficient: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.arch = arch
        self.num_classes = num_classes
        self.use_condition = use_condition
        self.use_gating = use_gating
        self.use_sparse_skip = use_sparse_skip
        self.gradient_checkpointing = gradient_checkpointing
        self.memory_efficient = memory_efficient
        
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
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    def enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        self.gradient_checkpointing = True
    
    def enable_memory_efficient_mode(self):
        """启用显存效率模式"""
        self.memory_efficient = True
        self.gradient_checkpointing = True
    
    def forward(self, inputs: torch.Tensor, data_samples: Optional[List] = None, mode: str = 'tensor', **kwargs):
        """Forward method."""
        if mode == 'tensor':
            return self.forward_tensor(inputs, data_samples)
        elif mode == 'predict':
            return self.forward_predict(inputs, data_samples)
        elif mode == 'loss':
            return self.forward_loss(inputs, data_samples)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward_tensor(self, inputs: torch.Tensor, data_samples: Optional[List] = None) -> torch.Tensor:
        """Forward pass - 显存优化版本"""
        
        # 优化的天气标签提取
        weather_label = self._extract_weather_labels_optimized(inputs, data_samples)
        
        # 使用梯度检查点或正常前向传播
        if self.gradient_checkpointing and self.training:
            # 梯度检查点版本
            encoder_outputs = checkpoint(
                self.encoder, inputs, weather_label, use_reentrant=False
            )
            decoder_output = checkpoint(
                self.decoder, encoder_outputs, use_reentrant=False
            )
        else:
            # 正常前向传播
            encoder_outputs = self.encoder(inputs, weather_label)
            decoder_output = self.decoder(encoder_outputs)
        
        logits = self.decode_head(decoder_output)
        
        # 显存效率模式：及时释放中间变量
        if self.memory_efficient and self.training:
            del encoder_outputs, decoder_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return logits
    
    def _extract_weather_labels_optimized(self, inputs: torch.Tensor, data_samples: Optional[List]) -> Optional[torch.Tensor]:
        """优化的天气标签提取 - 减少显存分配"""
        
        if not self.use_condition or data_samples is None or len(data_samples) == 0:
            return None
        
        batch_size = inputs.shape[0]
        
        # 直接创建张量，避免中间列表
        weather_labels = torch.zeros(batch_size, dtype=torch.long, device=inputs.device)
        
        for i in range(min(batch_size, len(data_samples))):
            sample = data_samples[i]
            weather = 0  # Default: clear weather
            
            # 提取天气标签
            try:
                if hasattr(sample, 'metainfo') and sample.metainfo is not None:
                    weather = sample.metainfo.get('weather_label', 0)
                elif hasattr(sample, 'weather_label'):
                    weather = sample.weather_label
                elif isinstance(sample, dict):
                    weather = sample.get('weather_label', 0)
                
                weather = int(weather) if weather is not None else 0
                weather = max(0, min(4, weather))  # Clamp to [0, 4]
            except:
                weather = 0
            
            weather_labels[i] = weather
        
        return weather_labels
    
    def forward_predict(self, inputs: torch.Tensor, data_samples: Optional[List] = None) -> List[Dict]:
        """Forward pass for prediction."""
        with torch.no_grad():  # 推理时禁用梯度
            logits = self.forward_tensor(inputs, data_samples)
            pred_label = logits.argmax(dim=1)
        
        predictions = []
        for i in range(pred_label.shape[0]):
            predictions.append({'segmentation': pred_label[i]})
        
        return predictions
    
    def forward_loss(self, inputs: torch.Tensor, data_samples: List) -> Dict[str, torch.Tensor]:
        """Forward pass for training with loss."""
        logits = self.forward_tensor(inputs, data_samples)
        
        # 优化的GT标签处理
        gt_labels = torch.stack([
            sample.gt_sem_seg.data.squeeze() if sample.gt_sem_seg.data.dim() > 2 
            else sample.gt_sem_seg.data
            for sample in data_samples
        ])

        loss = self.criterion(logits, gt_labels)
        
        return {'loss_seg': loss}