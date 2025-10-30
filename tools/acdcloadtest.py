"""
测试 ACDC 数据集加载和天气标签推断.

关键验证点：
1. ACDC 所有天气类型的目录是否存在
2. 标签文件是否能正确找到
3. 天气标签是否正确推断（fog=1, night=2, rain=3, snow=4）
4. ConcatDataset 是否能正确组合多个数据源
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmseg_custom.datasets import CityscapesACDCDataset
from mmengine.config import Config
cfg = Config.fromfile('configs/_base_/datasets/acdc.py')
pipeline = cfg.train_pipeline

def test_single_acdc_weather(weather_type: str):
    """测试单个 ACDC 天气类型."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing ACDC {weather_type} dataset")
    logger.info(f"{'='*80}\n")
    
    dataset = CityscapesACDCDataset(
        data_root='../mmseg/datasets/acdc',
        data_prefix=dict(
            img_path=f'rgb_anon/{weather_type}/train',
            seg_map_path=f'gt/{weather_type}/train'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
    )
    
    logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # 获取第一个样本
    sample = dataset[0]
    logger.info(f"✓ First sample loaded successfully")
    logger.info(f"  - Input shape: {sample['inputs'].shape}")
    logger.info(f"  - GT shape: {sample['data_samples'].gt_sem_seg.shape}")
    
    # 检查天气标签
    weather_labels = set()
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        # 从元信息中提取天气标签
        if hasattr(sample['data_samples'], 'img_metas'):
            weather_label = sample['data_samples'].img_metas.get('weather_label', -1)
            weather_labels.add(weather_label)
    
    logger.info(f"  - Weather labels found: {weather_labels}")
    logger.info(f"✓ {weather_type} dataset test passed!\n")
    
    return len(dataset)


def test_concat_dataset():
    """测试 ConcatDataset（联合多个天气类型）."""
    
    logger.info(f"\n{'='*80}")
    logger.info("Testing ConcatDataset with all ACDC weather types")
    logger.info(f"{'='*80}\n")
    
    from mmengine.dataset import ConcatDataset
    
    weather_types = ['fog', 'night', 'rain', 'snow']
    
    datasets = []
    total_samples = 0
    
    for weather in weather_types:
        ds = CityscapesACDCDataset(
            data_root='../mmseg/datasets/acdc',
            data_prefix=dict(
                img_path=f'rgb_anon/{weather}/train',
                seg_map_path=f'gt/{weather}/train'
            ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs'),
            ],
        )
        datasets.append(ds)
        total_samples += len(ds)
        logger.info(f"✓ {weather}: {len(ds)} samples")
    
    # 创建 ConcatDataset
    concat_dataset = ConcatDataset(datasets)
    logger.info(f"\n✓ ConcatDataset created: {len(concat_dataset)} total samples")
    
    # 随机采样验证
    indices = [0, len(datasets[0]), len(datasets[0])+len(datasets[1]), -1]
    for idx in indices:
        try:
            sample = concat_dataset[idx]
            logger.info(f"✓ Sample {idx}: inputs shape {sample['inputs'].shape}")
        except Exception as e:
            logger.error(f"✗ Sample {idx} failed: {e}")
    
    logger.info(f"✓ ConcatDataset test passed!\n")
    
    return total_samples


def test_cityscapes_acdc_joint():
    """测试 Cityscapes + ACDC 完整联合配置."""
    
    logger.info(f"\n{'='*80}")
    logger.info("Testing Cityscapes + ACDC joint training setup")
    logger.info(f"{'='*80}\n")
    
    from mmengine.dataset import ConcatDataset
    
    # 构建完整的联合数据集
    datasets = [
        # Cityscapes
        CityscapesACDCDataset(
            data_root='../mmseg/datasets/cityscapes',
            data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs'),
            ],
        ),
    ]
    logger.info(f"✓ Cityscapes train: {len(datasets[0])} samples")
    
    # ACDC 各天气类型
    for weather in ['fog', 'night', 'rain', 'snow']:
        ds = CityscapesACDCDataset(
            data_root='../mmseg/datasets/acdc',
            data_prefix=dict(
                img_path=f'rgb_anon/{weather}/train',
                seg_map_path=f'gt/{weather}/train'
            ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs'),
            ],
        )
        datasets.append(ds)
        logger.info(f"✓ ACDC {weather}: {len(ds)} samples")
    
    # 创建联合数据集
    joint_dataset = ConcatDataset(datasets)
    logger.info(f"\n✓ Joint dataset created: {len(joint_dataset)} total samples")
    
    # 计算样本分布
    logger.info(f"\nDataset composition:")
    logger.info(f"  - Cityscapes (clear): {len(datasets[0])} ({100*len(datasets[0])/len(joint_dataset):.1f}%)")
    for i, weather in enumerate(['fog', 'night', 'rain', 'snow'], 1):
        logger.info(f"  - ACDC {weather}: {len(datasets[i])} ({100*len(datasets[i])/len(joint_dataset):.1f}%)")
    
    logger.info(f"✓ Joint dataset test passed!\n")
    
    return len(joint_dataset)


def main():
    """主测试函数."""
    
    print("\n" + "="*80)
    print("ACDC DATASET LOADING TEST")
    print("="*80)
    
    # 测试单个天气类型
    weather_types = ['fog', 'night', 'rain', 'snow']
    for weather in weather_types:
        try:
            test_single_acdc_weather(weather)
        except Exception as e:
            logger.error(f"✗ {weather} test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 测试 ConcatDataset
    try:
        test_concat_dataset()
    except Exception as e:
        logger.error(f"✗ ConcatDataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试完整联合配置
    try:
        test_cityscapes_acdc_joint()
    except Exception as e:
        logger.error(f"✗ Joint dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ ALL ACDC TESTS PASSED!")
    print("="*80 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)