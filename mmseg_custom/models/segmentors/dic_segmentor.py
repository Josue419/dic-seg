"""
Complete DiC Segmentation Model integrating Encoder + Decoder + Head.

This is the main model interface for MMSegmentation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from mmengine.model import BaseModel
from mmseg.registry import MODELS  # ✅ 关键导入
from mmseg.models import build_backbone, build_head, build_loss

from ..backbones.dic_encoder import DicEncoder
from ..backbones.dic_decoder import DicDecoder


@MODELS.register_module()  # ✅ 关键：必须有这个装饰器！
class DicSegmentor(BaseModel):
    """
    Complete DiC Semantic Segmentation Model.
    
    Args:
        arch (str): Architecture size: 'S', 'B', 'XL'. Default: 'S'.
        num_classes (int): Number of segmentation classes. Default: 19 (Cityscapes).
        use_gating (bool): Enable conditional gating. Default: True.
        use_condition (bool): Enable weather condition input. Default: True.
        use_sparse_skip (bool): Use sparse stage-level skip. Default: True.
        num_weather_classes (int): Number of weather classes. Default: 5.
        with_auxiliary_head (bool): Whether to add auxiliary head on decoder output.
        loss_decode (dict): Loss function config for decoder output.
        auxiliary_head (dict, optional): Auxiliary head config.
        init_cfg (dict, optional): Weight initialization config.
    """
    
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
        
        # Segmentation head: 1x1 Conv + classification
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
        """
        Args:
            inputs (torch.Tensor): Image tensor, shape [B, 3, H, W].
            data_samples (List, optional): Data samples for training/val.
            mode (str): Forward mode: 'tensor', 'predict', 'loss'.
        
        Returns:
            Depends on mode.
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
        data_samples: Optional[List] = None,
    ) -> torch.Tensor:
        """Forward pass returning segmentation logits."""
        weather_label = None
        if data_samples is not None and len(data_samples) > 0:
            if hasattr(data_samples[0], 'metainfo') and 'weather_label' in data_samples[0].metainfo:
                weather_label_list = [
                    sample.metainfo['weather_label'] for sample in data_samples
                ]
                weather_label = torch.tensor(
                    weather_label_list, dtype=torch.long, device=inputs.device
                )
        
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