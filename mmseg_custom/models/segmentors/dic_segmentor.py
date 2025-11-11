"""
DiC Segmentation Model - 修复版本

关键修复：
1. 使用 PixelData 而非 InstanceData（MMSeg 要求）
2. 正确的预测格式转换
3. 完整的 forward_loss 实现
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, List, Dict
from mmengine.model import BaseModel
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.registry import MODELS

from ..backbones.dic_encoder import DicEncoder
from ..backbones.dic_decoder import DicDecoder


@MODELS.register_module()
class DicSegmentor(BaseModel):
    """DiC Semantic Segmentation Model - 完整修复版"""
    
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
        
        # Segmentation head: 1×1 卷积预测逻辑
        d0_out_channels = self.decoder.stage_channels[0]
        self.decode_head = nn.Conv2d(
            d0_out_channels, 
            num_classes, 
            kernel_size=1, 
            bias=True
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
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        data_samples: Optional[List[SegDataSample]] = None, 
        mode: str = 'tensor', 
        **kwargs
    ):
        """Forward method.
        
        Args:
            inputs: Input images, shape [B, 3, H, W]
            data_samples: List of SegDataSample objects
            mode: 'tensor' (forward), 'predict' (inference), 'loss' (training)
        
        Returns:
            Tensor or List[Dict] or Dict[str, Tensor]
        """
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
        data_samples: Optional[List[SegDataSample]] = None
    ) -> torch.Tensor:
        """Forward pass - 返回原始 logits
        
        Args:
            inputs: [B, 3, H, W]
            data_samples: Optional, 用于提取天气标签
        
        Returns:
            logits: [B, num_classes, H, W]
        """
        # 安全提取天气标签
        weather_label = self._extract_weather_labels(inputs, data_samples)
        
        # 前向传播（可选梯度检查点）
        if self.gradient_checkpointing and self.training:
            # 使用梯度检查点减少显存
            encoder_outputs = checkpoint(
                self.encoder, 
                inputs, 
                weather_label,
                use_reentrant=False
            )
            decoder_output = checkpoint(
                self.decoder, 
                encoder_outputs,
                use_reentrant=False
            )
        else:
            # 正常前向传播
            encoder_outputs = self.encoder(inputs, weather_label)
            decoder_output = self.decoder(encoder_outputs)
        
        # 分割头
        logits = self.decode_head(decoder_output)
        
        # 显存效率模式：及时释放中间变量
        if self.memory_efficient and self.training:
            del encoder_outputs, decoder_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return logits
    
    def forward_predict(
        self, 
        inputs: torch.Tensor, 
        data_samples: Optional[List[SegDataSample]] = None
    ) -> List[SegDataSample]:
        """Forward pass for inference.
        
        Args:
            inputs: [B, 3, H, W]
            data_samples: Optional, 将被填充预测结果
        
        Returns:
            List[SegDataSample]: 包含预测结果的数据样本列表
        """
        with torch.no_grad():
            logits = self.forward_tensor(inputs, data_samples)
            
            # 获取预测标签
            pred_label = logits.argmax(dim=1)  # [B, H, W]
        
        # 构造输出 SegDataSample
        if data_samples is None:
            data_samples = [
                SegDataSample() for _ in range(inputs.shape[0])
            ]
        
        # 填充预测结果到每个样本
        for i, sample in enumerate(data_samples):
            # ✅ 关键修复：使用 PixelData 而非 InstanceData
            sample.pred_sem_seg = PixelData(data=pred_label[i:i+1])
        
        return data_samples
    
    def forward_loss(
        self, 
        inputs: torch.Tensor, 
        data_samples: List[SegDataSample]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training with loss.
        
        Args:
            inputs: [B, 3, H, W]
            data_samples: List[SegDataSample] with gt_sem_seg
        
        Returns:
            Dict with 'loss_seg' key
        """
        # 前向传播获取 logits
        logits = self.forward_tensor(inputs, data_samples)
        
        # ✅ 关键修复：提取 GT 标签
        batch_size = len(data_samples)
        gt_labels_list = []
        
        for i in range(batch_size):
            sample = data_samples[i]
            
            # 获取 GT 标签张量
            if hasattr(sample, 'gt_sem_seg') and sample.gt_sem_seg is not None:
                gt_data = sample.gt_sem_seg.data  # [1, H, W] 或 [H, W]
                
                # 如果是 [1, H, W]，挤压为 [H, W]
                if gt_data.dim() == 3:
                    gt_data = gt_data.squeeze(0)
                
                gt_labels_list.append(gt_data)
            else:
                raise ValueError(f"Sample {i} missing gt_sem_seg")
        
        # 堆叠所有 GT 标签
        gt_labels = torch.stack(gt_labels_list)  # [B, H, W]
        
        # ✅ 计算损失
        loss = self.criterion(logits, gt_labels)
        
        # ✅ 关键修复：在训练时也要添加预测结果到 data_samples
        # 这样评估器才能在验证阶段取到 'pred_sem_seg'
        with torch.no_grad():
            pred_label = logits.argmax(dim=1)  # [B, H, W]
            
            for i, sample in enumerate(data_samples):
                # ✅ 使用 PixelData（MMSeg 标准格式）
                sample.pred_sem_seg = PixelData(data=pred_label[i:i+1])
        
        return {'loss_seg': loss}
    
    def _extract_weather_labels(
        self, 
        inputs: torch.Tensor, 
        data_samples: Optional[List[SegDataSample]]
    ) -> Optional[torch.Tensor]:
        """安全提取天气标签
        
        Args:
            inputs: [B, 3, H, W]
            data_samples: Optional list of SegDataSample
        
        Returns:
            weather_label: [B] tensor with values 0-4, or None
        """
        if not self.use_condition or data_samples is None:
            return None
        
        batch_size = inputs.shape[0]
        
        # 初始化为 0（clear weather）
        weather_labels = torch.zeros(
            batch_size, 
            dtype=torch.long, 
            device=inputs.device
        )
        
        # 从 data_samples 提取天气标签
        for i in range(min(batch_size, len(data_samples))):
            sample = data_samples[i]
            weather = 0  # 默认值：晴天
            
            try:
                # 尝试从 metainfo 提取
                if hasattr(sample, 'metainfo') and sample.metainfo is not None:
                    weather = sample.metainfo.get('weather_label', 0)
                
                # 尝试直接属性
                elif hasattr(sample, 'weather_label'):
                    weather = sample.weather_label
                
                # 尝试字典方式
                elif isinstance(sample, dict):
                    weather = sample.get('weather_label', 0)
                
                # 类型转换与范围限制
                weather = int(weather) if weather is not None else 0
                weather = max(0, min(4, weather))  # Clamp to [0, 4]
                
            except Exception as e:
                # 提取失败时使用默认值
                weather = 0
            
            weather_labels[i] = weather
        
        return weather_labels