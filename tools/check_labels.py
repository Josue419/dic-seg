"""
标签验证工具 - 检查 Cityscapes 标签是否有问题
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_cityscapes_labels(data_root: str):
    """检查 Cityscapes 标签文件"""
    
    data_root = Path(data_root)
    seg_dir = data_root / 'gtFine' / 'train'
    
    print(f"检查标签目录: {seg_dir}")
    
    if not seg_dir.exists():
        print(f"❌ 目录不存在: {seg_dir}")
        return
    
    # 查找所有标签文件
    label_files = list(seg_dir.rglob('*_gtFine_labelIds.png'))
    print(f"找到 {len(label_files)} 个标签文件")
    
    if len(label_files) == 0:
        print("❌ 没有找到标签文件")
        return
    
    # 检查前几个文件
    for i, label_file in enumerate(label_files[:10]):
        print(f"\n检查文件 {i+1}: {label_file.name}")
        
        try:
            # 使用 PIL 加载
            label_img = Image.open(label_file)
            label_array = np.array(label_img)
            
            print(f"  形状: {label_array.shape}")
            print(f"  数据类型: {label_array.dtype}")
            
            # 检查标签值
            unique_values = np.unique(label_array)
            print(f"  唯一值: {unique_values}")
            
            # 检查是否有超出范围的值
            valid_values = set(range(19)) | {255}
            invalid_values = set(unique_values) - valid_values
            
            if invalid_values:
                print(f"  ❌ 无效值: {invalid_values}")
            else:
                print(f"  ✅ 标签值正常")
                
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")


def check_tensor_conversion():
    """检查张量转换是否正常"""
    
    # 模拟一个标签张量
    label = torch.randint(0, 19, (512, 512), dtype=torch.long)
    
    # 添加一些 ignore 标签
    label[label == 0] = 255  # 将一些像素设为 ignore
    
    print(f"模拟标签张量:")
    print(f"  形状: {label.shape}")
    print(f"  数据类型: {label.dtype}")
    print(f"  唯一值: {torch.unique(label)}")
    print(f"  最小值: {label.min()}")
    print(f"  最大值: {label.max()}")
    
    # 检查是否有无效值
    invalid_mask = (label >= 19) & (label != 255)
    if invalid_mask.any():
        print(f"  ❌ 发现 {invalid_mask.sum()} 个无效像素")
    else:
        print(f"  ✅ 标签张量正常")


if __name__ == '__main__':
    print("="*60)
    print("Cityscapes 标签验证工具")
    print("="*60)
    
    # 检查标签文件
    data_root = '/root/projects/mmseg/datasets/cityscapes'
    check_cityscapes_labels(data_root)
    
    print("\n" + "="*60)
    print("张量转换测试")
    print("="*60)
    
    # 检查张量转换
    check_tensor_conversion()