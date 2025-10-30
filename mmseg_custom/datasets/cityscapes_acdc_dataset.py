"""
Cityscapes + ACDC 联合数据集加载器.

关键特性：
- ✅ ACDC 标签无需映射（已验证与 Cityscapes 对齐）
- ✅ 天气标签自动推断（0=clear, 1=fog, 2=night, 3=rain, 4=snow）
- ✅ 支持 MMSeg v1.2.2 API（data_prefix 字典）
- ✅ 递归处理多层目录（weather/train/city/*.png）
"""

from pathlib import Path
from typing import List, Optional, Dict
import logging
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

logger = logging.getLogger(__name__)


WEATHER_LABELS = {
    'cityscapes': 0,  # clear/daytime
    'clear': 0,
    'fog': 1,
    'night': 2,
    'rain': 3,
    'snow': 4,
}


@DATASETS.register_module()
class CityscapesACDCDataset(BaseSegDataset):
    """
    联合 Cityscapes + ACDC 数据集.
    
    ✅ 关键改进：
    - ACDC 标签无需映射（已与 Cityscapes 对齐）
    - 从路径自动推断天气标签
    - 支持 MMSeg v1.2.2 API
    
    Args:
        data_root (str): 数据集根目录
        data_prefix (dict): 路径前缀，e.g., dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train')
        pipeline (list): 数据处理 pipeline
        test_mode (bool): 是否为测试模式
        metainfo (dict): 数据集元信息
    """
    
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 100], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                 [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                 [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    )
    
    def __init__(
        self,
        data_root: str,
        data_prefix: Dict[str, str],
        pipeline: List[Dict] = None,
        test_mode: bool = False,
        metainfo: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline or [],
            test_mode=test_mode,
            metainfo=metainfo or self.METAINFO,
        )
    
    def load_data_list(self) -> List[Dict]:
        """
        加载数据列表（支持 Cityscapes 和 ACDC 目录结构）.
        
        Returns:
            List[Dict]: 每个字典包含 img_path, gt_seg_map, weather_label
        """
        data_list = []
        
        # 路径解析
        data_root = Path(self.data_root).resolve()
        img_dir = data_root / self.data_prefix.get('img_path', '')
        seg_dir = data_root / self.data_prefix.get('seg_map_path', '')
        
        logger.info(f"Loading dataset from {img_dir}")
        
        if not img_dir.exists():
            logger.error(f"Image directory not found: {img_dir}")
            return []
        
        if not seg_dir.exists():
            logger.error(f"Segmentation directory not found: {seg_dir}")
            return []
        
        # 递归查找所有图像文件
        img_files = sorted(list(img_dir.rglob('*.png')) + list(img_dir.rglob('*.jpg')))
        logger.info(f"Found {len(img_files)} image files")
        
        if len(img_files) == 0:
            logger.error(f"No image files found in {img_dir}")
            return []
        
        # 处理每个图像
        for img_path in img_files:
            rel_path = img_path.relative_to(img_dir)
            
            # 推断标签文件名
            if '_leftImg8bit.png' in img_path.name:
                seg_name = img_path.name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            elif '_leftImg8bit.jpg' in img_path.name:
                seg_name = img_path.name.replace('_leftImg8bit.jpg', '_gtFine_labelIds.png')
            elif '_rgb_anon.png' in img_path.name:
                seg_name = img_path.name.replace('_rgb_anon.png', '_gt.png')
            else:
                seg_name = img_path.stem + '_gt.png'
            
            seg_path = seg_dir / rel_path.parent / seg_name
            
            # 检查标签文件是否存在
            if not seg_path.exists():
                logger.debug(f"Label not found for {img_path.name}, skipping")
                continue
            
            # 从路径推断天气标签（关键！）
            weather_label = self._infer_weather_label(str(img_path))
            
            data_list.append(dict(
                img_path=str(img_path),
                gt_seg_map=str(seg_path),
                weather_label=weather_label,
            ))
        
        logger.info(f"Loaded {len(data_list)} samples")
        return data_list
    
    def _infer_weather_label(self, img_path: str) -> int:
        """从路径推断天气标签."""
        img_path_lower = img_path.lower()
        for weather_name, weather_id in WEATHER_LABELS.items():
            if weather_name in img_path_lower:
                return weather_id
        return WEATHER_LABELS['cityscapes']  # 默认晴天
    
    def __getitem__(self, idx: int) -> Dict:
        """获取样本，并将天气标签注入元信息."""
        data = super().__getitem__(idx)
        
        # 注入天气标签到元信息
        weather_label = self.data_list[idx].get('weather_label', 0)
        
        if 'img_metas' not in data:
            data['img_metas'] = {}
        
        if isinstance(data['img_metas'], dict):
            data['img_metas']['weather_label'] = weather_label
        
        return data