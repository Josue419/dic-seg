"""
论文级别的可视化工具集

功能模块：
1. SegmentationVisualizer - 分割结果可视化与对比
2. MetricsPlotter - 精度指标柱状图与曲线
3. ComparisonAnalyzer - 模型间对比分析
4. WeatherAnalyzer - 天气鲁棒性分析
5. FailureAnalyzer - 失败案例分析
"""

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import seaborn as sns
from PIL import Image
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 常量与配置
# ============================================================================

# Cityscapes 19 类定义
CITYSCAPES_CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
)

# 官方 Cityscapes 调色板
CITYSCAPES_PALETTE = [
    [128, 64, 128],    # road
    [244, 35, 232],    # sidewalk
    [70, 70, 70],      # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170, 100],   # traffic light
    [220, 220, 0],     # traffic sign
    [107, 142, 35],    # vegetation
    [152, 251, 152],   # terrain
    [70, 130, 180],    # sky
    [220, 20, 60],     # person
    [255, 0, 0],       # rider
    [0, 0, 142],       # car
    [0, 0, 70],        # truck
    [0, 60, 100],      # bus
    [0, 80, 100],      # train
    [0, 0, 230],       # motorcycle
    [119, 11, 32],     # bicycle
]

WEATHER_NAMES = ['clear', 'fog', 'night', 'rain', 'snow']
WEATHER_COLORS = ['#2ecc71', '#95a5a6', '#34495e', '#3498db', '#f39c12']


# ============================================================================
# 1. 分割结果可视化
# ============================================================================

class SegmentationVisualizer:
    """
    分割结果可视化工具
    
    生成：
    - 预测彩色可视化
    - 预测与 GT 对比图
    - 错误热力图
    - 透明度融合效果
    """
    
    def __init__(self, predictions_dir: Path, gt_dir: Optional[Path] = None):
        """
        Args:
            predictions_dir: 预测结果存储目录（npz 或 png）
            gt_dir: 真值标签目录（可选）
        """
        self.predictions_dir = Path(predictions_dir)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        
        # 创建调色板映射
        self.palette_array = np.array(CITYSCAPES_PALETTE, dtype=np.uint8)
        
        logger.info(f"SegmentationVisualizer initialized")
        logger.info(f"  Predictions: {self.predictions_dir}")
        logger.info(f"  Ground truth: {self.gt_dir}")
    
    def label_to_rgb(self, label: np.ndarray) -> np.ndarray:
        """将标签图转换为 RGB 彩色图"""
        rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        
        for class_id in range(len(CITYSCAPES_CLASSES)):
            mask = label == class_id
            rgb[mask] = self.palette_array[class_id]
        
        # 处理 ignore_index (255)
        if (label == 255).any():
            rgb[label == 255] = [128, 128, 128]  # 灰色
        
        return rgb
    
    def create_overlay(
        self,
        image: np.ndarray,
        label: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        创建标签与图像的透明度融合
        
        Args:
            image: 原始 RGB 图像
            label: 标签图
            alpha: 标签的透明度 (0=纯图像, 1=纯标签)
        """
        label_rgb = self.label_to_rgb(label)
        overlay = (image * (1 - alpha) + label_rgb * alpha).astype(np.uint8)
        return overlay
    
    def create_error_map(
        self,
        pred: np.ndarray,
        gt: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        创建错误热力图
        
        Returns:
            error_map: 错误热力图 (255=正确, 0=错误)
            accuracy: 像素级准确率
        """
        error_map = (pred == gt).astype(np.uint8) * 255
        accuracy = error_map.mean() / 255.0
        return error_map, accuracy
    
    def save_comparison_grid(
        self,
        image_path: Path,
        pred: np.ndarray,
        gt: Optional[np.ndarray],
        output_path: Path,
        dpi: int = 150
    ):
        """
        保存对比网格图（原图 | 预测 | GT | 错误）
        """
        image = np.array(Image.open(image_path))
        pred_rgb = self.label_to_rgb(pred)
        
        fig, axes = plt.subplots(1, 4 if gt is not None else 3, figsize=(16, 4), dpi=dpi)
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 预测
        axes[1].imshow(pred_rgb)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        if gt is not None:
            # 真值
            gt_rgb = self.label_to_rgb(gt)
            axes[2].imshow(gt_rgb)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
            
            # 错误热力图
            error_map, accuracy = self.create_error_map(pred, gt)
            axes[3].imshow(error_map, cmap='RdYlGn')
            axes[3].set_title(f'Errors (Acc: {accuracy:.2%})')
            axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison grid: {output_path}")
    
    def save_overlay_images(
        self,
        output_dir: Path,
        alpha: float = 0.6,
        num_samples: int = 50
    ):
        """
        批量生成透明度融合图像
        
        Args:
            output_dir: 输出目录
            alpha: 透明度
            num_samples: 生成的样本数
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集所有预测文件
        pred_files = sorted(list(self.predictions_dir.glob('*.png')) + 
                           list(self.predictions_dir.glob('*.npz')))[:num_samples]
        
        logger.info(f"Generating {len(pred_files)} overlay images...")
        
        for pred_file in pred_files:
            # 加载预测
            if pred_file.suffix == '.npz':
                data = np.load(pred_file)
                pred = data['prediction']
            else:
                pred = np.array(Image.open(pred_file))
            
            # 加载原始图像（假设同名 jpg）
            img_file = pred_file.with_suffix('.jpg')
            if not img_file.exists():
                img_file = pred_file.with_stem(pred_file.stem.replace('_pred', '')).with_suffix('.jpg')
            
            if img_file.exists():
                image = np.array(Image.open(img_file))
                overlay = self.create_overlay(image, pred, alpha)
                
                output_file = output_dir / pred_file.with_stem(pred_file.stem + '_overlay').name
                Image.fromarray(overlay).save(output_file)
        
        logger.info(f"✅ Overlay images saved to {output_dir}")


# ============================================================================
# 2. 精度指标可视化
# ============================================================================

class MetricsPlotter:
    """
    精度指标可视化
    
    生成：
    - 各类 mIoU 柱状图
    - 精度曲线（训练/验证）
    - 混淆矩阵热力图
    """
    
    def __init__(self, metrics_dir: Path):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_class_miou(
        self,
        class_miou: Dict[str, float],
        output_path: Path,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        绘制各类 mIoU 柱状图
        
        Args:
            class_miou: {'class_name': mIoU_value, ...}
            output_path: 保存路径
        """
        classes = list(class_miou.keys())
        values = list(class_miou.values())
        
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        # 按数值排序
        sorted_items = sorted(zip(classes, values), key=lambda x: x[1])
        classes, values = zip(*sorted_items)
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(classes)))
        bars = ax.barh(classes, values, color=colors)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.01, i, f'{val:.2%}', va='center', fontsize=9)
        
        ax.set_xlabel('mIoU', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Semantic IoU', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved class mIoU chart: {output_path}")
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        output_path: Path,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        绘制混淆矩阵热力图
        
        Args:
            confusion_matrix: [num_classes, num_classes]
            output_path: 保存路径
        """
        # 归一化
        cm_norm = confusion_matrix.astype(float) / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        sns.heatmap(
            cm_norm,
            annot=False,  # 不显示数值（太密集）
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Normalized Count'},
            xticklabels=CITYSCAPES_CLASSES,
            yticklabels=CITYSCAPES_CLASSES,
            ax=ax
        )
        
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ground Truth Class', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix: {output_path}")
    
    def plot_training_curves(
        self,
        log_data: Dict[str, List[float]],
        output_path: Path,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """
        绘制训练曲线（loss / mIoU）
        
        Args:
            log_data: {
                'train_loss': [...],
                'val_loss': [...],
                'val_miou': [...]
            }
            output_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)
        
        # Loss 曲线
        if 'train_loss' in log_data:
            axes[0].plot(log_data['train_loss'], label='Train', linewidth=2, marker='o', markersize=3)
        if 'val_loss' in log_data:
            axes[0].plot(log_data['val_loss'], label='Validation', linewidth=2, marker='s', markersize=3)
        
        axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
        axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # mIoU 曲线
        if 'val_miou' in log_data:
            axes[1].plot(log_data['val_miou'], label='mIoU', linewidth=2, marker='o', color='#e74c3c')
        
        axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('mIoU', fontsize=11, fontweight='bold')
        axes[1].set_title('Validation mIoU', fontsize=12, fontweight='bold')
        axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves: {output_path}")


# ============================================================================
# 3. 对比分析
# ============================================================================

class ComparisonAnalyzer:
    """
    模型间对比分析
    
    生成：
    - 模型精度对比（柱状 + 折线）
    - 参数量/FLOPs/速度对比
    - Pareto 前沿分析（精度 vs 效率）
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'miou',
        output_path: Optional[Path] = None
    ):
        """
        绘制模型间精度对比
        
        Args:
            results: {
                'DiC-S': {'miou': 0.78, 'dataset': 'cityscapes'},
                'U-Net': {'miou': 0.72, 'dataset': 'cityscapes'},
                ...
            }
            metric: 比较指标 ('miou', 'fps', 'params')
            output_path: 保存路径
        """
        models = list(results.keys())
        values = [results[m][metric] for m in models]
        
        output_path = output_path or self.output_dir / f'comparison_{metric}.png'
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        colors = ['#e74c3c' if 'DiC' in m else '#3498db' for m in models]
        bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison: {metric.upper()}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison chart: {output_path}")
    
    def plot_pareto_frontier(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: Optional[Path] = None
    ):
        """
        绘制 Pareto 前沿（精度 vs 参数量）
        
        Args:
            results: {
                'DiC-S': {'miou': 0.78, 'params': 50},  # M
                ...
            }
        """
        output_path = output_path or self.output_dir / 'pareto_frontier.png'
        
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        
        for model_name, metrics in results.items():
            params = metrics.get('params', 0)
            miou = metrics.get('miou', 0)
            
            color = '#e74c3c' if 'DiC' in model_name else '#3498db'
            size = 300 if 'DiC' in model_name else 150
            
            ax.scatter(params, miou, s=size, alpha=0.6, color=color, edgecolor='black', linewidth=1)
            ax.annotate(model_name, (params, miou), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Parameters (Million)', fontsize=12, fontweight='bold')
        ax.set_ylabel('mIoU', fontsize=12, fontweight='bold')
        ax.set_title('Pareto Frontier: Accuracy vs Model Size', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Pareto frontier: {output_path}")


# ============================================================================
# 4. 天气鲁棒性分析
# ============================================================================

class WeatherAnalyzer:
    """
    天气鲁棒性分析
    
    生成：
    - 各天气条件下的精度对比
    - 鲁棒性评分
    - 降级曲线（性能 vs 天气恶劣程度）
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_weather_performance(
        self,
        weather_results: Dict[str, Dict[str, float]],
        output_path: Optional[Path] = None
    ):
        """
        绘制各天气条件下的性能对比
        
        Args:
            weather_results: {
                'clear': {'miou': 0.82, 'fps': 25},
                'fog': {'miou': 0.75, 'fps': 24},
                'night': {'miou': 0.68, 'fps': 24},
                'rain': {'miou': 0.70, 'fps': 24},
                'snow': {'miou': 0.65, 'fps': 24}
            }
        """
        output_path = output_path or self.output_dir / 'weather_performance.png'
        
        weathers = list(weather_results.keys())
        miou_values = [weather_results[w]['miou'] for w in weathers]
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        bars = ax.bar(weathers, miou_values, 
                     color=WEATHER_COLORS[:len(weathers)],
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # 添加数值标签
        for bar, val in zip(bars, miou_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('mIoU', fontsize=12, fontweight='bold')
        ax.set_title('Performance Across Weather Conditions', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(miou_values) * 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved weather performance chart: {output_path}")
    
    def plot_robustness_score(
        self,
        results_by_weather: Dict[str, float],
        baseline_clear_miou: float,
        output_path: Optional[Path] = None
    ):
        """
        绘制鲁棒性评分
        
        Robustness = 平均恶劣天气 mIoU / 晴天 mIoU
        
        Args:
            results_by_weather: {'clear': 0.82, 'fog': 0.75, ...}
            baseline_clear_miou: 晴天基准 mIoU
        """
        output_path = output_path or self.output_dir / 'robustness_score.png'
        
        challenging_weathers = [w for w in WEATHER_NAMES[1:] if w in results_by_weather]
        challenging_mious = [results_by_weather[w] for w in challenging_weathers]
        
        avg_challenging_miou = np.mean(challenging_mious)
        robustness_score = avg_challenging_miou / baseline_clear_miou
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        
        # 柱状图
        weathers = ['clear'] + challenging_weathers
        mious = [baseline_clear_miou] + challenging_mious
        colors = WEATHER_COLORS
        
        bars = ax1.bar(weathers, mious, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.axhline(avg_challenging_miou, color='red', linestyle='--', linewidth=2, label=f'Avg challenging: {avg_challenging_miou:.2%}')
        ax1.set_ylabel('mIoU', fontsize=11, fontweight='bold')
        ax1.set_title('mIoU by Weather Condition', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 鲁棒性计分卡
        ax2.text(0.5, 0.6, f'{robustness_score:.2%}', 
                ha='center', va='center', fontsize=48, fontweight='bold',
                transform=ax2.transAxes)
        ax2.text(0.5, 0.3, 'Robustness Score', 
                ha='center', va='center', fontsize=16, fontweight='bold',
                transform=ax2.transAxes)
        ax2.text(0.5, 0.1, f'(avg. adverse / clear weather)', 
                ha='center', va='center', fontsize=10, style='italic',
                transform=ax2.transAxes)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved robustness score: {output_path}")
        return robustness_score


# ============================================================================
# 5. 失败案例分析
# ============================================================================

class FailureAnalyzer:
    """
    失败案例分析
    
    生成：
    - 最易混淆类对排行
    - 难以分割的场景统计
    - 失败案例可视化
    """
    
    def __init__(self, predictions_dir: Path, gt_dir: Path, output_dir: Path):
        self.predictions_dir = Path(predictions_dir)
        self.gt_dir = Path(gt_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.palette_array = np.array(CITYSCAPES_PALETTE, dtype=np.uint8)
    
    def find_failure_cases(
        self,
        num_cases: int = 20,
        metric: str = 'iou'  # 'iou' 或 'accuracy'
    ) -> List[Tuple[str, float]]:
        """
        找出表现最差的样本
        
        Returns:
            [(image_name, error_value), ...]，按错误从小到大排序
        """
        failure_cases = []
        
        pred_files = sorted(self.predictions_dir.glob('*.npz'))
        
        for pred_file in pred_files:
            # 加载预测
            data = np.load(pred_file)
            pred = data['prediction']
            
            # 加载 GT
            gt_file = self.gt_dir / pred_file.with_suffix('.png').name
            if not gt_file.exists():
                continue
            gt = np.array(Image.open(gt_file))
            
            # 计算指标
            if metric == 'iou':
                # 计算每类 IoU
                ious = []
                for cls_id in range(19):
                    pred_mask = (pred == cls_id)
                    gt_mask = (gt == cls_id)
                    
                    intersection = np.sum(pred_mask & gt_mask)
                    union = np.sum(pred_mask | gt_mask)
                    
                    iou = intersection / (union + 1e-8)
                    ious.append(iou)
                
                error_value = np.mean(ious)
            
            else:  # accuracy
                error_value = np.mean(pred == gt)
            
            failure_cases.append((pred_file.stem, error_value))
        
        # 按错误升序排序（最差的在前）
        failure_cases.sort(key=lambda x: x[1])
        
        return failure_cases[:num_cases]
    
    def visualize_failure_cases(
        self,
        failure_cases: List[Tuple[str, float]],
        images_dir: Path,
        num_display: int = 10
    ):
        """
        可视化失败案例
        """
        num_display = min(num_display, len(failure_cases))
        
        fig, axes = plt.subplots(num_display, 3, figsize=(12, 3*num_display), dpi=150)
        
        if num_display == 1:
            axes = axes.reshape(1, -1)
        
        for row, (case_name, error_value) in enumerate(failure_cases[:num_display]):
            # 加载图像、预测、GT
            img_file = images_dir / f"{case_name}.jpg"
            if not img_file.exists():
                continue
            
            pred_file = self.predictions_dir / f"{case_name}.npz"
            gt_file = self.gt_dir / f"{case_name}.png"
            
            image = np.array(Image.open(img_file))
            pred = np.load(pred_file)['prediction']
            gt = np.array(Image.open(gt_file))
            
            # 绘制
            axes[row, 0].imshow(image)
            axes[row, 0].set_title('Image')
            axes[row, 0].axis('off')
            
            pred_rgb = np.zeros_like(image)
            for cls_id in range(19):
                pred_rgb[pred == cls_id] = self.palette_array[cls_id]
            axes[row, 1].imshow(pred_rgb)
            axes[row, 1].set_title(f'Pred (mIoU: {error_value:.2%})')
            axes[row, 1].axis('off')
            
            gt_rgb = np.zeros_like(image)
            for cls_id in range(19):
                gt_rgb[gt == cls_id] = self.palette_array[cls_id]
            axes[row, 2].imshow(gt_rgb)
            axes[row, 2].set_title('GT')
            axes[row, 2].axis('off')
        
        plt.tight_layout()
        output_file = self.output_dir / 'failure_cases.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved failure cases visualization: {output_file}")
    
    def plot_confusion_top_pairs(
        self,
        confusion_matrix: np.ndarray,
        top_k: int = 10,
        output_path: Optional[Path] = None
    ):
        """
        绘制最易混淆的类对
        
        Args:
            confusion_matrix: [19, 19]
            top_k: 显示前 k 对
        """
        output_path = output_path or self.output_dir / 'confusion_top_pairs.png'
        
        # 找出最大的非对角线元素
        confusion_norm = confusion_matrix.astype(float) / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        pairs = []
        for i in range(len(CITYSCAPES_CLASSES)):
            for j in range(len(CITYSCAPES_CLASSES)):
                if i != j:
                    pairs.append((CITYSCAPES_CLASSES[i], CITYSCAPES_CLASSES[j], 
                                 confusion_norm[i, j]))
        
        # 排序
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 绘制
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        labels = [f"{p[0]} → {p[1]}" for p in pairs[:top_k]]
        values = [p[2] for p in pairs[:top_k]]
        
        bars = ax.barh(labels, values, color='#e74c3c', edgecolor='black', linewidth=1)
        
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.2%}', va='center', fontsize=9)
        
        ax.set_xlabel('Confusion Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_k} Most Confused Class Pairs', fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(values) * 1.15)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion pairs chart: {output_path}")


# ============================================================================
# 工具函数
# ============================================================================

def generate_paper_figures(
    predictions_dir: Path,
    gt_dir: Path,
    images_dir: Path,
    output_base_dir: Path,
    metrics_json: Optional[Path] = None
):
    """
    一键生成所有论文图表
    
    使用示例：
        generate_paper_figures(
            predictions_dir=Path('work_dirs/dic_s/predictions'),
            gt_dir=Path('data/cityscapes/gtFine/val'),
            images_dir=Path('data/cityscapes/leftImg8bit/val'),
            output_base_dir=Path('paper_figures'),
            metrics_json=Path('work_dirs/dic_s/results.json')
        )
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 开始生成论文图表...")
    
    # 1. 分割可视化
    logger.info("📊 生成分割结果可视化...")
    viz = SegmentationVisualizer(predictions_dir, gt_dir)
    viz.save_overlay_images(output_base_dir / 'overlay_images', num_samples=50)
    
    # 2. 指标可视化
    logger.info("📈 生成指标图表...")
    plotter = MetricsPlotter(output_base_dir / 'metrics')
    
    if metrics_json and metrics_json.exists():
        with open(metrics_json, 'r') as f:
            metrics = json.load(f)
        
        if 'class_miou' in metrics:
            plotter.plot_class_miou(metrics['class_miou'], 
                                   output_base_dir / 'metrics' / 'class_miou.png')
    
    # 3. 失败案例分析
    logger.info("❌ 生成失败案例分析...")
    analyzer = FailureAnalyzer(predictions_dir, gt_dir, output_base_dir / 'failures')
    failure_cases = analyzer.find_failure_cases(num_cases=20)
    
    logger.info(f"✅ 论文图表生成完成！")
    logger.info(f"   输出目录: {output_base_dir}")