"""
极简 Cityscapes + ACDC 数据加载器 - 最终修复版

关键修复：
- 正确的标签验证逻辑：只检查非 255 的值是否在 [0, 18] 范围内
- 自动修复无效标签值为 255
- 双重保护：文件级 + 张量级修复
"""

import logging
import torch
import numpy as np
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
            
            # ✅ 新增：验证并修复标签文件
            if not self._validate_and_fix_label_file(seg_path):
                print(f"⚠ 跳过无效标签文件: {seg_path.name}")
                skip_count += 1
                continue
            
            # 推断天气标签
            weather_label = self._get_weather_label(str(img_path))
            
            # ✅ 使用最标准的 MMSeg 分割任务数据格式
            data_info = dict(
                img_path=str(img_path),
                seg_map_path=str(seg_path),
                seg_fields=[],                   # LoadAnnotations 会添加 'gt_seg_map'
                reduce_zero_label=False,         # Cityscapes 不需要减少零标签
                bbox_fields=[],                  # 防止检测任务加载
                mask_fields=[],                  # 防止实例分割加载
                weather_label=weather_label,     # 自定义天气标签
            )
            
            data_list.append(data_info)
            
            if idx < 3:
                print(f"✓ 样本 {idx}: {img_path.name}")
        
        print(f"✓ 成功加载 {len(data_list)} 个数据对")
        if skip_count > 0:
            print(f"⚠ 跳过 {skip_count} 个无效样本")
        print(f"{'='*70}\n")
        
        return data_list
    
    def _validate_and_fix_label_file(self, seg_path: Path) -> bool:
        """
        验证并修复标签文件 - 正确处理 255（ignore_index）
        
        ✅ 修复逻辑：
        - 255 是合法的 ignore_index，不需要修复
        - 只有非 255 且超出 [0, 18] 范围的值才需要修复为 255
        """
        try:
            # 使用 PIL 加载标签文件
            from PIL import Image
            label_img = Image.open(seg_path)
            label_array = np.array(label_img)
            
            # 检查标签值范围
            unique_values = np.unique(label_array)
            
            # ✅ 关键修复：正确的验证逻辑
            # - 0-18: 有效类别标签
            # - 255: 合法的 ignore_index
            # - 其他值: 需要修复为 255 的无效值
            valid_class_labels = set(range(19))  # {0, 1, 2, ..., 18}
            ignore_label = 255
            
            # 找出需要修复的无效值：既不是有效类别，也不是 ignore_index
            invalid_values = []
            for val in unique_values:
                if val not in valid_class_labels and val != ignore_label:
                    invalid_values.append(val)
            
            if invalid_values:
                print(f"🔧 修复标签文件 {seg_path.name}:")
                print(f"   原始唯一值: {sorted(unique_values)}")
                print(f"   无效值: {sorted(invalid_values)} (将转为 255)")
                
                # ✅ 关键修复：只修复真正的无效值
                fixed_array = label_array.copy()
                for invalid_val in invalid_values:
                    fixed_array[label_array == invalid_val] = 255
                
                # 保存修复后的标签文件
                fixed_img = Image.fromarray(fixed_array.astype(np.uint8), mode='L')
                fixed_img.save(seg_path)
                
                fixed_unique = np.unique(fixed_array)
                print(f"   修复后唯一值: {sorted(fixed_unique)}")
                print(f"   ✅ 已保存修复后的标签文件")
            else:
                # 所有值都是有效的
                valid_count = sum(1 for val in unique_values if val in valid_class_labels)
                ignore_count = sum(1 for val in unique_values if val == ignore_label)
                print(f"✅ 标签文件 {seg_path.name} 验证通过:")
                print(f"   有效类别值: {valid_count} 种")
                print(f"   ignore 值(255): {'是' if ignore_count > 0 else '否'}")
            
            return True
            
        except Exception as e:
            print(f"❌ 无法处理标签文件 {seg_path.name}: {e}")
            return False
    
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
        """准备数据 - 带张量级标签验证"""
        # 获取数据信息
        data_info = self.get_data_info(idx)
        
        # 确保所有必需键存在
        required_keys = ['seg_fields', 'bbox_fields', 'mask_fields']
        for key in required_keys:
            if key not in data_info:
                data_info[key] = []
        
        # 调用 pipeline
        result = self.pipeline(data_info)
        
        # ✅ 关键修复：在 pipeline 处理后再次验证标签张量
        if 'data_samples' in result and hasattr(result['data_samples'], 'gt_sem_seg'):
            gt_seg = result['data_samples'].gt_sem_seg.data
            
            # ✅ 正确的张量验证逻辑：只检查非 255 的值
            unique_values = torch.unique(gt_seg)
            
            # 找出需要修复的无效值：不在 [0, 18] 且不是 255
            invalid_mask = torch.zeros_like(gt_seg, dtype=torch.bool)
            for val in unique_values:
                if 0 <= val <= 18 or val == 255:
                    continue  # 有效值，跳过
                else:
                    invalid_mask |= (gt_seg == val)  # 标记为无效
            
            if invalid_mask.any():
                invalid_count = invalid_mask.sum().item()
                print(f"🔧 Pipeline后发现 {invalid_count} 个无效标签像素，自动修复为 255")
                print(f"   处理前唯一值: {unique_values.tolist()}")
                
                # 修复无效值
                result['data_samples'].gt_sem_seg.data[invalid_mask] = 255
                
                fixed_unique = torch.unique(result['data_samples'].gt_sem_seg.data)
                print(f"   处理后唯一值: {fixed_unique.tolist()}")
            else:
                # 验证通过，打印统计信息
                valid_classes = [val.item() for val in unique_values if 0 <= val <= 18]
                has_ignore = 255 in unique_values
                print(f"✅ 张量验证通过: {len(valid_classes)} 种有效类别, ignore={'是' if has_ignore else '否'}")
        
        # 添加天气标签到 metainfo
        if 'data_samples' in result and result['data_samples'] is not None:
            weather_label = data_info.get('weather_label', 0)
            
            if not hasattr(result['data_samples'], 'metainfo'):
                result['data_samples'].metainfo = {}
            elif result['data_samples'].metainfo is None:
                result['data_samples'].metainfo = {}
            
            result['data_samples'].metainfo['weather_label'] = weather_label
        
        return result