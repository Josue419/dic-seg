"""
DiC-S 生产训练配置：Cityscapes Only

用途：在 Cityscapes 上进行完整 200 epoch 训练（基准实验）

继承：
- 模型：DiC-S
- 数据集：Cityscapes 仅
- 优化器：AdamW (lr=6e-5)
- 学习率：PolyLR (200 epochs)
- 运行时：标准配置
"""

_base_ = [
    '_base_/models/dic_s.py',
    '_base_/datasets/cityscapes.py',
    '_base_/schedules/poly_200e.py',
    '_base_/default_runtime.py',
]

# ============================================================================
# 生产参数（无需覆盖，使用 _base_ 默认值）
# ============================================================================

# 评估：每 5 个 epoch 验证一次
evaluation = dict(interval=5, metric='mIoU', save_best='mIoU')

# ============================================================================
# 工作目录
# ============================================================================

work_dir = './work_dirs/dic_s_cityscapes'
exp_name = 'dic_s_cityscapes'