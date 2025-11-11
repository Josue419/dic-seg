"""
DiC-S 联合训练配置：Cityscapes + ACDC 完整训练

用途：训练天气鲁棒分割模型，包含所有天气条件

继承：
- 模型：DiC-S (启用条件机制)
- 数据集：Cityscapes + ACDC 联合
- 优化器：AdamW (lr=6e-5)
- 学习率：PolyLR (200 epochs)
- 运行时：标准配置

数据组成：
- Cityscapes train: ~2975 samples (clear)
- ACDC fog train: ~400 samples
- ACDC night train: ~400 samples  
- ACDC rain train: ~400 samples
- ACDC snow train: ~400 samples
- 总计: ~4575 samples/epoch
"""

_base_ = [
    '_base_/models/dic_s.py',
    '_base_/datasets/cityscapes.py',
    '_base_/schedules/poly_50e.py',
    '_base_/default_runtime.py',
]
custom_imports = dict(
    imports=['mmseg_custom','mmengine.dataset'],  
    allow_failed_imports=False
)
# ============================================================================
# 模型配置覆盖 - 启用所有条件机制
# ============================================================================

model = dict(
    type='DicSegmentor',
    arch='S',
    num_classes=19,
    use_gating=True,        # 启用条件门控
    use_condition=True,     # 启用天气条件
    use_sparse_skip=True,   # 启用稀疏跳跃连接
    num_weather_classes=5,  # 5种天气类型
    gradient_checkpointing=True,  # ✅ 启用梯度检查点
)

# ============================================================================
# 训练配置优化
# ============================================================================

# 联合训练需要更多 epoch 收敛
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)



# 数据加载优化
train_dataloader = dict(
    batch_size=4,  # 根据显存调整
    num_workers=4,
    persistent_workers=True,
    drop_last=True,  # 多卡必需：确保 batch 能整除 GPU 数
    pin_memory=True,
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
)

# ============================================================================
# 评估与保存策略
# ============================================================================

# 更频繁的评估
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=5,  # 每 5 epoch 保存
        max_keep_ckpts=2,
        save_best='val/mIoU',
        save_last=True,
    ),
)

'''
继续训练配置
resume_from = '/path/to/your/checkpoint.pth'  # 你之前保存的 checkpoint
resume = True  # 启用 resume 模式
'''

'''
 ✅ 只加载模型权重，重置优化器和 epoch
load_from = 'work_dirs/dic_s_cs/best_val_mIoU_epoch_18.pth'  # 你之前保存的 checkpoint
resume = False  # 不启用 resume 模式（默认值）
'''

# ============================================================================
# 工作目录
# ============================================================================

work_dir = './work_dirs/dic_s_cs'
exp_name = 'dics_acdc'