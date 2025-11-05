"""
完整版 Cityscapes + ACDC 数据加载器 - 支持预处理标签

关键功能：
- 自动使用预处理标签（.processed_labels 目录）
- 支持 Cityscapes 和 ACDC 格式
- 自动推断天气标签
- 生产就绪的性能优化
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

logger = logging.getLogger(__name__)


@DATASETS.register_module()
class CityscapesACDCDataset(BaseSegDataset):
    """完整版 Cityscapes + ACDC 数据加载器"""
    
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
    
    def __init__(self, use_processed_labels: bool = True, **kwargs):
        """
        Args:
            use_processed_labels: 是否使用预处理的标签文件
        """
        # ✅ 关键修复：属性赋值 BEFORE super().__init__()
        self.use_processed_labels = use_processed_labels
        
        # 预先准备 processed_root（如果需要）
        if use_processed_labels and 'data_root' in kwargs:
            data_root = Path(kwargs['data_root'])
            self.processed_root = data_root / '.processed_labels'
        else:
            self.processed_root = None
        
        # 现在调用 super().__init__()，它会立即调用 load_data_list()
        # 此时 self.processed_root 已经存在
        super().__init__(**kwargs)
        
        # 验证预处理状态（可选的后初始化）
        if use_processed_labels and self.processed_root:
            if not self._check_preprocessing_completed():
                logger.warning("数据集可能未完成预处理，建议运行 tools/prep_labels.py")
    
    def _check_preprocessing_completed(self) -> bool:
        """检查数据集是否已完成预处理"""
        if not self.processed_root:
            return False
        metadata_file = self.processed_root / 'preprocessing_metadata.json'
        return metadata_file.exists()
    
    def _get_processed_label_path(self, original_path: Path) -> Path:
        """获取预处理标签路径"""
        if not self.use_processed_labels or not self.processed_root:
            return original_path
        
        try:
            rel_path = original_path.relative_to(Path(self.data_root))
        except ValueError:
            rel_path = Path(original_path.name)
        
        processed_path = self.processed_root / rel_path
        
        # 如果预处理文件存在，使用预处理版本
        if processed_path.exists():
            return processed_path
        else:
            logger.warning(f"预处理标签不存在，使用原始标签: {original_path.name}")
            return original_path
    
    
    def load_data_list(self) -> List[Dict]:
        """加载数据列表 - 支持预处理标签"""
        
        if isinstance(self.data_root, str):
            data_root = Path(self.data_root)
        else:
            data_root = self.data_root
        
        if not data_root.is_absolute():
            data_root = data_root.resolve()
        
        img_dir = data_root / self.data_prefix['img_path']
        seg_dir = data_root / self.data_prefix['seg_map_path']
        
        logger.info(f"Loading dataset from {data_root}")
        logger.info(f"  Image dir: {img_dir}")
        logger.info(f"  Segmentation dir: {seg_dir}")
        logger.info(f"  Use processed labels: {self.use_processed_labels}")
        
        if not img_dir.exists() or not seg_dir.exists():
            logger.error(f"Directory not found: {img_dir} or {seg_dir}")
            return []
        
        # 查找所有图像文件
        img_files = sorted(
            list(img_dir.rglob('*.png')) + 
            list(img_dir.rglob('*.jpg'))
        )
        
        if len(img_files) == 0:
            logger.error(f"No image files found in {img_dir}")
            return []
        
        logger.info(f"Found {len(img_files)} images")
        
        # 配对图像和标签
        data_list = []
        skip_count = 0
        
        for img_path in img_files:
            rel_path = img_path.relative_to(img_dir)
            
            # 推断标签文件名 - 支持 Cityscapes 和 ACDC 格式
            if '_leftImg8bit.png' in img_path.name:
                # Cityscapes 格式
                seg_name = img_path.name.replace(
                    '_leftImg8bit.png',
                    '_gtFine_labelIds.png'
                )
            elif '_rgb_anon.png' in img_path.name:
                # ACDC 格式
                seg_name = img_path.name.replace('_rgb_anon.png', '_gt_labelIds.png')
            else:
                # 通用格式
                seg_name = img_path.stem + '_gt.png'
            
            original_seg_path = seg_dir / rel_path.parent / seg_name
            
            if not original_seg_path.exists():
                skip_count += 1
                continue
            
            # 获取最终的标签路径（可能是预处理版本）
            final_seg_path = self._get_processed_label_path(original_seg_path)
            
            # 推断天气标签
            weather_label = self._get_weather_label(str(img_path))
            
            data_info = dict(
                img_path=str(img_path),
                seg_map_path=str(final_seg_path),  # 使用最终路径
                seg_fields=[],
                reduce_zero_label=False,
                bbox_fields=[],
                mask_fields=[],
                weather_label=weather_label,
            )
            
            data_list.append(data_info)
        
        logger.info(f"Loaded {len(data_list)} samples")
        if skip_count > 0:
            logger.warning(f"Skipped {skip_count} samples due to missing labels")
        
        # 统计天气标签分布
        weather_dist = self._get_weather_distribution(data_list)
        logger.info(f"Weather distribution: {weather_dist}")
        
        return data_list
    
    def _get_weather_label(self, path: str) -> int:
        """推断天气标签"""
        path_lower = path.lower()
        
        if 'fog' in path_lower:
            return 1
        elif 'night' in path_lower:
            return 2
        elif 'rain' in path_lower:
            return 3
        elif 'snow' in path_lower:
            return 4
        else:
            return 0  # clear
    
    def _get_weather_distribution(self, data_list: List[Dict]) -> Dict[str, int]:
        """统计天气标签分布"""
        weather_names = ['clear', 'fog', 'night', 'rain', 'snow']
        distribution = {name: 0 for name in weather_names}
        
        for data_info in data_list:
            weather_idx = data_info['weather_label']
            if 0 <= weather_idx < len(weather_names):
                distribution[weather_names[weather_idx]] += 1
        
        return distribution
    
    def prepare_data(self, idx: int) -> Dict:
        """准备数据"""
        # 获取数据信息
        data_info = self.get_data_info(idx)
        
        # 确保必需键存在
        required_keys = ['seg_fields', 'bbox_fields', 'mask_fields']
        for key in required_keys:
            if key not in data_info:
                data_info[key] = []
        
        # 应用 pipeline
        result = self.pipeline(data_info)
        
        # 添加天气标签到 metainfo
        if 'data_samples' in result and result['data_samples'] is not None:
            weather_label = data_info.get('weather_label', 0)
            
            if not hasattr(result['data_samples'], 'metainfo'):
                result['data_samples'].metainfo = {}
            elif result['data_samples'].metainfo is None:
                result['data_samples'].metainfo = {}
            
            result['data_samples'].metainfo['weather_label'] = weather_label
        
        return result