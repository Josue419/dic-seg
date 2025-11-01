"""
DiC-S 联合训练调试配置（适合RTX 4050）

修改点：
1. 减小batch_size和num_workers
2. 限制训练epochs
3. 完整的数据归一化
"""

_base_ = [
    '_base_/models/dic_s.py',
    '_base_/datasets/acdc.py',
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
    use_condition=False,     # 启用天气条件
    use_sparse_skip=False,   # 启用稀疏跳跃连接
    num_weather_classes=5,  # 5种天气类型
)
# RTX 4050 优化配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2)

# 学习率调度适配
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=2,
        by_epoch=True,
    )
]

# 数据加载：适合4050的配置
train_dataloader = dict(
    batch_size=1,        # 4050显存安全值
    num_workers=0,       # 调试时设为0，避免多进程问题
    persistent_workers=False,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
)

# 评估：每epoch验证
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')

work_dir = './work_dirs/dic_s_cityscapes_acdc_debug'
exp_name = 'dics_4050'