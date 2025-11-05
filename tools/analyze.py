# tools/analyze.py
from pathlib import Path
from mmseg_custom.utils.visualization import generate_paper_figures

# 一键生成所有论文图表
generate_paper_figures(
    predictions_dir=Path('work_dirs/dic_s/outputs/predictions'),
    gt_dir=Path('data/cityscapes/gtFine/val'),
    images_dir=Path('data/cityscapes/leftImg8bit/val'),
    output_base_dir=Path('paper_figures'),
    metrics_json=Path('work_dirs/dic_s/results.json')
)

"""
单独使用各个可视化器

from pathlib import Path
from mmseg_custom.utils.visualization import (
    SegmentationVisualizer,
    WeatherAnalyzer,
    MetricsPlotter
)

# 分割结果可视化
viz = SegmentationVisualizer(
    predictions_dir=Path('predictions'),
    gt_dir=Path('gt')
)
viz.save_overlay_images(Path('output/overlays'))

# 天气鲁棒性分析
weather_analyzer = WeatherAnalyzer(Path('output/weather'))
weather_analyzer.plot_weather_performance({
    'clear': {'miou': 0.82, 'fps': 25},
    'fog': {'miou': 0.75, 'fps': 24},
    'night': {'miou': 0.68, 'fps': 24},
    'rain': {'miou': 0.70, 'fps': 24},
    'snow': {'miou': 0.65, 'fps': 24}
})

# 精度指标
plotter = MetricsPlotter(Path('output/metrics'))
plotter.plot_training_curves({
    'train_loss': [0.5, 0.3, 0.2],
    'val_loss': [0.6, 0.35, 0.25],
    'val_miou': [0.65, 0.75, 0.80]
})
"""