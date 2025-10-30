"""
极简 Cityscapes + ACDC 数据加载器 - 最终完全修复版

关键改进：
- 在 load_data_list 中添加 seg_fields 和 reduce_zero_label 键
- 这是 MMSeg v1.2.2 LoadAnnotations 所需的标准键
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
        """加载数据列表"""
        
        if isinstance(self.data_root, str):
            data_root = Path(self.data_root)
        else:
            data_root = self.data_root
        
        if not data_root.is_absolute():
            data_root = data_root.resolve()
        
        img_dir = data_root / self.data_prefix['img_path']
        seg_dir = data_root / self.data_prefix['seg_map_path']
        
        print(f"\n{'='*70}")
        print(f"[DataLoader Debug] 数据集加载信息")
        print(f"{'='*70}")
        print(f"  data_root (原始):  {self.data_root}")
        print(f"  data_root (绝对):  {data_root}")
        print(f"  img_prefix:        {self.data_prefix['img_path']}")
        print(f"  seg_prefix:        {self.data_prefix['seg_map_path']}")
        print(f"  img_dir (计算):    {img_dir}")
        print(f"  seg_dir (计算):    {seg_dir}")
        print(f"  data_root 存在:    {data_root.exists()}")
        print(f"  img_dir 存在:      {img_dir.exists()}")
        print(f"  seg_dir 存在:      {seg_dir.exists()}")
        
        # 检查目录
        if not data_root.exists():
            print(f"\n❌ 数据根目录不存在！")
            print(f"   期望路径: {data_root}")
            print(f"{'='*70}\n")
            return []
        
        if not img_dir.exists():
            print(f"\n❌ 图像目录不存在！")
            print(f"   期望路径: {img_dir}")
            print(f"   data_root 下的目录: {list(data_root.iterdir())}")
            print(f"{'='*70}\n")
            return []
        
        if not seg_dir.exists():
            print(f"\n❌ 标签目录不存在！")
            print(f"   期望路径: {seg_dir}")
            print(f"   data_root 下的目录: {list(data_root.iterdir())}")
            print(f"{'='*70}\n")
            return []
        
        # 查找图像文件
        img_files = sorted(
            list(img_dir.rglob('*.png')) + 
            list(img_dir.rglob('*.jpg'))
        )
        
        print(f"✓ 找到 {len(img_files)} 个图像文件")
        
        if len(img_files) == 0:
            print(f"\n❌ 没有找到任何图像文件！")
            print(f"   搜索路径: {img_dir}")
            print(f"{'='*70}\n")
            return []
        
        # 配对图像和标签
        data_list = []
        skip_count = 0
        
        for idx, img_path in enumerate(img_files):
            rel_path = img_path.relative_to(img_dir)
            
            # 推断标签文件名
            if '_leftImg8bit.png' in img_path.name:
                seg_name = img_path.name.replace(
                    '_leftImg8bit.png',
                    '_gtFine_labelIds.png'
                )
            elif '_leftImg8bit.jpg' in img_path.name:
                seg_name = img_path.name.replace(
                    '_leftImg8bit.jpg',
                    '_gtFine_labelIds.png'
                )
            elif '_rgb_anon.png' in img_path.name:
                seg_name = img_path.name.replace('_rgb_anon.png', '_gt.png')
            else:
                seg_name = img_path.stem + '_gt.png'
            
            seg_path = seg_dir / rel_path.parent / seg_name
            
            if not seg_path.exists():
                skip_count += 1
                if idx < 5:
                    print(f"  ⚠ 标签缺失: {rel_path} → {seg_name}")
                continue
            
            # 推断天气标签
            weather_label = self._get_weather_label(str(img_path))
            
            # ✅ 关键改进：添加所有必需的键
            # - seg_map_path: 标签文件路径
            # - seg_fields: MMSeg v1.2.2 要求的分割字段名列表
            # - reduce_zero_label: 是否需要减少零标签（Cityscapes 不需要）
            # - weather_label: 天气标签（自定义）
            data_list.append(dict(
                img_path=str(img_path),
                seg_map_path=str(seg_path),
                seg_fields=['seg_map_path'],  # ✅ 添加这一行
                reduce_zero_label=False,  # ✅ 添加这一行
                weather_label=weather_label,
            ))
        
        print(f"✓ 成功配对 {len(data_list)} 个数据对")
        if skip_count > 0:
            print(f"⚠ 跳过 {skip_count} 个缺失标签的图像")
        
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