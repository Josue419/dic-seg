"""
DiC-S 联合训练配置：Cityscapes + ACDC（所有天气）

用途：训练天气鲁棒分割模型

继承：
- 模型：DiC-S
- 数据集：Cityscapes + ACDC (fog/night/rain/snow)
- 优化器：AdamW (lr=6e-5)
- 学习率：PolyLR (200 epochs)
- 运行时：标准配置

数据组成：
- Cityscapes train: ~2975 samples (clear)
- ACDC fog: ~500 samples
- ACDC night: ~500 samples
- ACDC rain: ~500 samples
- ACDC snow: ~500 samples
- 总计: ~5000+ samples/epoch
"""

_base_ = [
    '_base_/models/dic_s.py',
    '_base_/datasets/cityscapes_acdc.py',
    '_base_/schedules/poly_200e.py',
    '_base_/default_runtime.py',
]

# ============================================================================
# 联合训练参数（无需覆盖，使用 _base_ 默认值）
# ============================================================================

# 评估：每 5 个 epoch 验证一次
evaluation = dict(interval=5, metric='mIoU', save_best='mIoU')

# ============================================================================
# 工作目录
# ============================================================================

work_dir = './work_dirs/dic_s_cityscapes_acdc'
exp_name = 'dic_s_cityscapes_acdc'