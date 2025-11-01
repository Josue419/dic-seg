"""
一次性标签预处理脚本 - 修复标签值并存储到单独目录

功能：
1. 检查 Cityscapes 和 ACDC 标签的有效性
2. 将非 0-18 的标签值转为 255（ignore_index）
3. 存储到 .processed_labels 目录，保持原文件不变
4. 生成预处理元数据和统计信息
"""

import argparse
import json
import time
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LabelPreprocessor:
    """标签预处理器 - 分离存储方案"""
    
    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)
        self.processed_root = self.dataset_root / '.processed_labels'
        self.processed_root.mkdir(exist_ok=True)
        
        self.metadata_file = self.processed_root / 'preprocessing_metadata.json'
    
    def create_dataset_fingerprint(self) -> str:
        """创建数据集指纹，检测是否需要重新预处理"""
        label_files = list(self.dataset_root.rglob('*_labelIds.png'))[:100]  # 采样前100个
        fingerprint_data = []
        
        for file_path in sorted(label_files):
            if file_path.exists():
                stat = file_path.stat()
                fingerprint_data.append(f"{file_path.name}:{stat.st_size}:{stat.st_mtime}")
        
        return hashlib.md5(''.join(fingerprint_data).encode()).hexdigest()
    
    def get_processed_label_path(self, original_path: Path) -> Path:
        """获取预处理后标签文件的存储路径"""
        try:
            rel_path = original_path.relative_to(self.dataset_root)
        except ValueError:
            rel_path = Path(original_path.name)
        
        processed_path = self.processed_root / rel_path
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        return processed_path
    
    def is_processed(self, original_path: Path) -> bool:
        """检查文件是否已被预处理"""
        processed_path = self.get_processed_label_path(original_path)
        
        if not processed_path.exists():
            return False
        
        # 检查文件是否比原文件新
        try:
            original_mtime = original_path.stat().st_mtime
            processed_mtime = processed_path.stat().st_mtime
            return processed_mtime >= original_mtime
        except:
            return False
    
    def preprocess_label_file(self, original_path: Path) -> Path:
        """预处理单个标签文件"""
        
        if self.is_processed(original_path):
            return self.get_processed_label_path(original_path)
        
        # 加载原始标签
        label_img = Image.open(original_path)
        label_array = np.array(label_img)
        
        # 检查并修复标签值
        unique_values = np.unique(label_array)
        valid_labels = set(range(19)) | {255}
        invalid_labels = set(unique_values) - valid_labels
        
        if invalid_labels:
            logger.info(f"🔧 处理 {original_path.name}: 修复 {len(invalid_labels)} 个无效标签")
            
            # 修复无效标签
            corrected_array = label_array.copy()
            for invalid_val in invalid_labels:
                corrected_array[label_array == invalid_val] = 255
        else:
            corrected_array = label_array
        
        # 保存到处理后的路径
        processed_path = self.get_processed_label_path(original_path)
        corrected_img = Image.fromarray(corrected_array.astype(np.uint8), mode='L')
        corrected_img.save(processed_path)
        
        return processed_path
    
    def batch_preprocess_cityscapes(self):
        """批量预处理 Cityscapes 标签"""
        label_files = []
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_root / 'gtFine' / split
            if split_dir.exists():
                label_files.extend(list(split_dir.rglob('*_gtFine_labelIds.png')))
        
        return self._process_files(label_files, "Cityscapes")
    
    def batch_preprocess_acdc(self):
        """批量预处理 ACDC 标签"""
        label_files = []
        for weather in ['fog', 'night', 'rain', 'snow']:
            for split in ['train', 'val']:
                split_dir = self.dataset_root / 'gt' / weather / split
                if split_dir.exists():
                    label_files.extend(list(split_dir.rglob('*_gt_labelIds.png')))
        
        return self._process_files(label_files, "ACDC")
    
    def _process_files(self, label_files: list, dataset_name: str):
        """处理文件列表"""
        logger.info(f"找到 {len(label_files)} 个 {dataset_name} 标签文件")
        
        stats = {
            'dataset_name': dataset_name,
            'total_files': len(label_files),
            'processed_files': 0,
            'skipped_files': 0,
            'invalid_labels_found': set(),
            'errors': 0,
            'fingerprint': self.create_dataset_fingerprint(),
            'timestamp': time.time()
        }
        
        for label_file in tqdm(label_files, desc=f"预处理 {dataset_name} 标签"):
            try:
                if self.is_processed(label_file):
                    stats['skipped_files'] += 1
                    continue
                
                self.preprocess_label_file(label_file)
                stats['processed_files'] += 1
                
            except Exception as e:
                logger.error(f"处理失败 {label_file}: {e}")
                stats['errors'] += 1
        
        # 保存元数据
        stats['invalid_labels_found'] = list(stats['invalid_labels_found'])
        self._save_metadata(stats)
        
        return stats
    
    def _save_metadata(self, stats: dict):
        """保存预处理元数据"""
        metadata = {
            'dataset_root': str(self.dataset_root),
            'processed_root': str(self.processed_root),
            'stats': stats,
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def preprocess_cityscapes(data_root: Path, force: bool = False):
    """预处理 Cityscapes 数据集"""
    preprocessor = LabelPreprocessor(data_root)
    
    # 检查是否需要重新处理
    if not force and preprocessor.metadata_file.exists():
        try:
            with open(preprocessor.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            current_fingerprint = preprocessor.create_dataset_fingerprint()
            if metadata.get('stats', {}).get('fingerprint') == current_fingerprint:
                logger.info("✅ Cityscapes 数据集已经预处理过，跳过")
                return metadata['stats']
        except:
            pass
    
    logger.info("🔄 开始预处理 Cityscapes 数据集...")
    stats = preprocessor.batch_preprocess_cityscapes()
    
    logger.info(f"✅ Cityscapes 预处理完成:")
    logger.info(f"   总文件: {stats['total_files']}")
    logger.info(f"   处理文件: {stats['processed_files']}")
    logger.info(f"   跳过文件: {stats['skipped_files']}")
    logger.info(f"   错误: {stats['errors']}")
    
    return stats


def preprocess_acdc(data_root: Path, force: bool = False):
    """预处理 ACDC 数据集"""
    preprocessor = LabelPreprocessor(data_root)
    
    # 检查是否需要重新处理
    if not force and preprocessor.metadata_file.exists():
        try:
            with open(preprocessor.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            current_fingerprint = preprocessor.create_dataset_fingerprint()
            if metadata.get('stats', {}).get('fingerprint') == current_fingerprint:
                logger.info("✅ ACDC 数据集已经预处理过，跳过")
                return metadata['stats']
        except:
            pass
    
    logger.info("🔄 开始预处理 ACDC 数据集...")
    stats = preprocessor.batch_preprocess_acdc()
    
    logger.info(f"✅ ACDC 预处理完成:")
    logger.info(f"   总文件: {stats['total_files']}")
    logger.info(f"   处理文件: {stats['processed_files']}")
    logger.info(f"   跳过文件: {stats['skipped_files']}")
    logger.info(f"   错误: {stats['errors']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='一次性预处理数据集标签')
    parser.add_argument('--cityscapes-root', type=str, 
                       default='/root/projects/mmseg/datasets/cityscapes',
                       help='Cityscapes 数据集根目录')
    parser.add_argument('--acdc-root', type=str,
                       default='/root/projects/mmseg/datasets/acdc', 
                       help='ACDC 数据集根目录')
    parser.add_argument('--force', action='store_true',
                       help='强制重新预处理')
    
    args = parser.parse_args()
    
    logger.info("🚀 开始一次性数据预处理...")
    
    # 预处理 Cityscapes
    if Path(args.cityscapes_root).exists():
        preprocess_cityscapes(Path(args.cityscapes_root), args.force)
    else:
        logger.warning(f"⚠ Cityscapes 路径不存在: {args.cityscapes_root}")
    
    # 预处理 ACDC  
    if Path(args.acdc_root).exists():
        preprocess_acdc(Path(args.acdc_root), args.force)
    else:
        logger.warning(f"⚠ ACDC 路径不存在: {args.acdc_root}")
    
    logger.info("✅ 所有数据预处理完成！")


if __name__ == '__main__':
    main()