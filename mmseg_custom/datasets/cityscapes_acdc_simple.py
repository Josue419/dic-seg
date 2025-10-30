"""
极简 Cityscapes + ACDC 数据加载器 - 彻底修复版

关键修复：
- 确保数据字典包含所有必需键
- 防止 mmcv LoadAnnotations 尝试加载 instances
- 正确传递天气标签
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

logger = logging.getLogger(__name__)


@DATASETS.register_module()
class CityscapesACDCSimple(BaseSegDataset):
    """极简 Cityscapes + ACDC 数据加载器"""
    
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
    
    def load_data_list(self) -> List[Dict]:
        """加载数据列表 - 确保与 MMSeg LoadAnnotations 完全兼容"""
        
        if isinstance(self.data_root, str):
            data_root = Path(self.data_root)
        else:
            data_root = self.data_root
        
        if not data_root.is_absolute():
            data_root = data_root.resolve()
        
        img_dir = data_root / self.data_prefix['img_path']
        seg_dir = data_root / self.data_prefix['seg_map_path']
        
        print(f"\n{'='*70}")
        print(f"[FIXED DataLoader Debug] 数据集加载信息")
        print(f"{'='*70}")
        print(f"  data_root: {data_root}")
        print(f"  img_dir: {img_dir}")
        print(f"  seg_dir: {seg_dir}")
        print(f"  img_dir 存在: {img_dir.exists()}")
        print(f"  seg_dir 存在: {seg_dir.exists()}")
        
        if not img_dir.exists() or not seg_dir.exists():
            print(f"❌ 目录不存在!")
            return []
        
        # 查找图像文件
        img_files = sorted(
            list(img_dir.rglob('*.png')) + 
            list(img_dir.rglob('*.jpg'))
        )
        
        print(f"✓ 找到 {len(img_files)} 个图像文件")
        
        if len(img_files) == 0:
            return []
        
        # 配对图像和标签
        data_list = []
        skip_count = 0
        
        for idx, img_path in enumerate(img_files):
            # 限制样本数量以便调试
            if idx >= 10:
                break
                
            rel_path = img_path.relative_to(img_dir)
            
            # 推断标签文件名
            if '_leftImg8bit.png' in img_path.name:
                seg_name = img_path.name.replace(
                    '_leftImg8bit.png',
                    '_gtFine_labelIds.png'
                )
            else:
                seg_name = img_path.stem + '_gt.png'
            
            seg_path = seg_dir / rel_path.parent / seg_name
            
            if not seg_path.exists():
                skip_count += 1
                continue
            
            # 推断天气标签
            weather_label = self._get_weather_label(str(img_path))
            
            # ✅ 关键修复：使用最标准的 MMSeg 分割任务数据格式
            data_info = dict(
                # 基本路径
                img_path=str(img_path),
                seg_map_path=str(seg_path),
                
                # ✅ 分割任务必需键
                seg_fields=[],                   # LoadAnnotations 会添加 'gt_seg_map'
                reduce_zero_label=False,         # Cityscapes 不需要减少零标签
                
                # ✅ 防止检测任务加载的键
                bbox_fields=[],
                mask_fields=[],
                
                # 自定义字段
                weather_label=weather_label,
            )
            
            data_list.append(data_info)
            
            if idx < 3:
                print(f"✓ 样本 {idx}: {img_path.name}")
        
        print(f"✓ 成功加载 {len(data_list)} 个数据对")
        if skip_count > 0:
            print(f"⚠ 跳过 {skip_count} 个缺失标签的样本")
        print(f"{'='*70}\n")
        
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
    
    def prepare_data(self, idx: int) -> Dict:
        """准备数据 - 确保所有必需键存在"""
        # 获取数据信息
        data_info = self.get_data_info(idx)
        
        # 确保所有必需键存在
        required_keys = ['seg_fields', 'bbox_fields', 'mask_fields']
        for key in required_keys:
            if key not in data_info:
                data_info[key] = []
        
        # 调用 pipeline
        result = self.pipeline(data_info)
        
        # 将天气标签添加到 data_samples.metainfo
        if 'data_samples' in result and result['data_samples'] is not None:
            weather_label = data_info.get('weather_label', 0)
            
            # 初始化 metainfo
            if not hasattr(result['data_samples'], 'metainfo'):
                result['data_samples'].metainfo = {}
            elif result['data_samples'].metainfo is None:
                result['data_samples'].metainfo = {}
            
            result['data_samples'].metainfo['weather_label'] = weather_label
        
        return result