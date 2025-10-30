"""
数据加载器调试工具 - 修复版本

修复问题：
1. 正确导入所有 mmseg transforms
2. 增强错误处理和调试信息
3. 逐步验证 pipeline 的每个步骤
"""

import argparse
import sys
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from torch.utils.data import DataLoader
from mmengine.dataset import default_collate
# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ✅ 关键修复：正确导入所有必需模块
import mmseg
import mmseg.datasets
import mmseg.models
import mmseg.datasets.transforms  # ✅ 确保 PackSegInputs 被注册

# 导入自定义模块
from mmseg_custom import *


def debug_single_sample(dataset, idx=0):
    """调试单个样本的数据结构"""
    print(f"\n{'='*60}")
    print(f"[DEBUG] 分析数据集样本 #{idx}")
    print(f"{'='*60}")
    
    try:
        # 获取原始数据信息
        data_info = dataset.get_data_info(idx)
        print(f"✓ 原始数据信息:")
        for key, value in data_info.items():
            if isinstance(value, str) and len(value) > 50:
                print(f"  {key}: {value[:50]}...")
            else:
                print(f"  {key}: {value}")
        
        # 检查必需的键
        required_keys = ['img_path', 'seg_map_path', 'seg_fields']
        for key in required_keys:
            if key not in data_info:
                print(f"  ❌ 缺少必需键: {key}")
                return False
            else:
                print(f"  ✅ 包含必需键: {key}")
        
        # 通过 pipeline 处理
        print(f"\n📥 通过 pipeline 处理...")
        processed_data = dataset[idx]
        
        print(f"✓ 处理后的数据结构:")
        for key, value in processed_data.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {type(value)} - shape: {value.shape}")
            elif hasattr(value, '__len__') and not isinstance(value, str):
                print(f"  {key}: {type(value)} - length: {len(value)}")
            else:
                print(f"  {key}: {type(value)}")
        
        # ✅ 关键检查：inputs 必须是 tensor
        if 'inputs' in processed_data:
            inputs = processed_data['inputs']
            print(f"\n🔍 关键检查 - inputs:")
            print(f"  类型: {type(inputs)}")
            if torch.is_tensor(inputs):
                print(f"  ✅ 是 torch.Tensor!")
                print(f"  形状: {inputs.shape}")
                print(f"  数据类型: {inputs.dtype}")
                print(f"  设备: {inputs.device}")
            else:
                print(f"  ❌ 不是 tensor! 实际类型: {type(inputs)}")
                if isinstance(inputs, list):
                    print(f"  列表长度: {len(inputs)}")
                    if inputs:
                        print(f"  第一个元素类型: {type(inputs[0])}")
                return False
        
        # 检查 data_samples
        if 'data_samples' in processed_data:
            data_samples = processed_data['data_samples']
            print(f"\n🔍 关键检查 - data_samples:")
            print(f"  类型: {type(data_samples)}")
            
            # 检查天气标签
            weather_label = None
            if hasattr(data_samples, 'metainfo') and data_samples.metainfo:
                weather_label = data_samples.metainfo.get('weather_label')
                print(f"  ✅ 天气标签: {weather_label}")
            else:
                print(f"  ⚠ 没有找到天气标签")
        
        return True
        
    except Exception as e:
        print(f"❌ 样本 #{idx} 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_pipeline_steps(cfg):
    """逐步调试 pipeline 中的每个 transform"""
    print(f"\n{'='*60}")
    print(f"[DEBUG] 逐步调试 Pipeline")
    print(f"{'='*60}")
    
    try:
        # 构建数据集（不使用 pipeline）
        train_dataset_cfg = cfg.train_dataloader.dataset.copy()
        train_dataset_cfg['pipeline'] = []  # 暂时清空 pipeline
        
        from mmseg.registry import DATASETS
        dataset = DATASETS.build(train_dataset_cfg)
        
        # 获取原始数据
        data_info = dataset.get_data_info(0)
        print(f"✓ 原始数据信息: {list(data_info.keys())}")
        
        # 检查必需键
        if 'seg_fields' not in data_info:
            print(f"❌ 原始数据缺少 seg_fields 键!")
            return False
        
        # 逐步应用每个 transform
        pipeline = cfg.train_dataloader.dataset.pipeline
        current_data = data_info.copy()
        
        from mmseg.registry import TRANSFORMS
        
        for step_idx, transform_cfg in enumerate(pipeline):
            transform_type = transform_cfg['type']
            print(f"\n🔄 步骤 {step_idx + 1}: {transform_type}")
            
            try:
                # 构建 transform
                transform = TRANSFORMS.build(transform_cfg)
                print(f"  ✅ Transform 构建成功: {type(transform)}")
                
                # 应用 transform
                current_data = transform(current_data)
                
                print(f"  ✅ Transform 应用成功")
                print(f"  输出键: {list(current_data.keys())}")
                
                # 检查关键字段
                for key in ['inputs', 'data_samples', 'img', 'gt_seg_map']:
                    if key in current_data:
                        value = current_data[key]
                        if hasattr(value, 'shape'):
                            print(f"    {key}: {type(value)} - shape: {value.shape}")
                        else:
                            print(f"    {key}: {type(value)}")
                
                # 特别检查 inputs 类型
                if 'inputs' in current_data:
                    inputs = current_data['inputs']
                    if torch.is_tensor(inputs):
                        print(f"    ✅ inputs 是 torch.Tensor")
                    else:
                        print(f"    ❌ inputs 不是 tensor: {type(inputs)}")
                
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\n✅ Pipeline 全部步骤执行成功!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_dataloader(cfg):
    """调试完整的 DataLoader"""
    print(f"\n{'='*60}")
    print(f"[DEBUG] 分析 DataLoader")
    print(f"{'='*60}")
    
    try:
        # 构建数据集
        train_dataset_cfg = cfg.train_dataloader.dataset.copy()
        
        from mmseg.registry import DATASETS
        dataset = DATASETS.build(train_dataset_cfg)
        
        print(f"✓ 数据集构建成功: {len(dataset)} 样本")
        
        # 调试单个样本
        if not debug_single_sample(dataset, idx=0):
            return False
        
        # 构建 DataLoader
        sampler = DefaultSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
        
            collate_fn=default_collate,
        )
        
        print(f"\n📦 DataLoader 构建成功")
        
        # 获取第一个 batch
        print(f"\n🔄 获取第一个 batch...")
        for batch_idx, batch_data in enumerate(dataloader):
            print(f"✓ Batch #{batch_idx} 获取成功")
            print(f"  Batch 类型: {type(batch_data)}")
            
            # 检查 inputs
            if 'inputs' in batch_data:
                inputs = batch_data['inputs']
                print(f"\n🎯 关键检查 - batch inputs:")
                print(f"    类型: {type(inputs)}")
                if torch.is_tensor(inputs):
                    print(f"    ✅ 是 tensor!")
                    print(f"    形状: {inputs.shape}")
                else:
                    print(f"    ❌ 不是 tensor!")
                    return False
            
            # 只测试第一个 batch
            break
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Debug data loading issues')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"数据加载器调试工具 - 修复版")
    print(f"{'='*80}")
    print(f"配置文件: {args.config}")
    
    # 加载配置
    try:
        cfg = Config.fromfile(args.config)
        print(f"✅ 配置文件加载成功")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False
    
    # 调试步骤
    success = True
    
    # 1. 逐步调试 pipeline
    print(f"\n📋 步骤 1: 逐步调试 Pipeline")
    if not debug_pipeline_steps(cfg):
        success = False
    
    # 2. 调试完整 DataLoader
    if success:
        print(f"\n📦 步骤 2: 调试完整 DataLoader")
        if not debug_dataloader(cfg):
            success = False
    
    # 总结
    print(f"\n{'='*80}")
    if success:
        print(f"✅ 所有调试步骤通过!")
        print(f"数据加载管道工作正常，inputs 正确转换为 torch.Tensor")
    else:
        print(f"❌ 发现问题!")
        print(f"请检查上述错误信息并修复")
    print(f"{'='*80}")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)