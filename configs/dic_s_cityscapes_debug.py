"""
DiC-S 调试配置（2 epochs）

用途：快速验证模型/数据加载/条件机制是否正常

继承：
- 模型：DiC-S
- 数据集：Cityscapes 仅
- 优化器：AdamW (lr=6e-5)
- 学习率：PolyLR (50 epochs，但被覆盖为 2)
- 运行时：标准配置
"""

_base_ = [
    '_base_/models/dic_s.py',
    '_base_/datasets/cityscapes.py',
    '_base_/schedules/poly_50e.py',
    '_base_/default_runtime.py',
]

# ============================================================================
# 调试参数覆盖
# ============================================================================

# 训练循环：2 epochs（快速验证）
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2)

# 数据加载：调试时减小 num_workers 和 batch_size
train_dataloader = dict(
    batch_size=1,
    num_workers=0,  # 调试时设为 0，避免多进程问题
    persistent_workers=False,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
)

# 评估：每 epoch 验证一次
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')

# ============================================================================
# 工作目录
# ============================================================================

work_dir = './work_dirs/dic_s_cityscapes_debug'
exp_name = 'dic_s_cityscapes_debug'