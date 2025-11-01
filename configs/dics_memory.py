"""
DiC-S 显存优化训练配置

关键优化：
1. 梯度检查点
2. 混合精度训练
3. 显存监控
4. 小 batch size + 梯度累积
"""

_base_ = [
    '_base_/models/dic_s.py',
    '_base_/datasets/cs_acdc.py',
    '_base_/schedules/poly_200e.py',
    '_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg_custom', 'mmengine.dataset'],
    allow_failed_imports=False
)

# 模型配置 - 启用显存优化
model = dict(
    type='DicSegmentor',
    arch='S',
    num_classes=19,
    use_gating=True,
    use_condition=True,
    use_sparse_skip=True,
    num_weather_classes=5,
    gradient_checkpointing=True,  # 启用梯度检查点
    memory_efficient=True,        # 启用显存效率模式
)

# 混合精度训练
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',  # 自动混合精度
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=1.0),  # 梯度裁剪
)

# 显存优化的数据加载配置
train_dataloader = dict(
    batch_size=1,  # 小 batch size
    num_workers=2,
    persistent_workers=True,
    pin_memory=False,  # 关闭 pin_memory 减少显存占用
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=False,
)
"""# 自定义 hooks
custom_hooks = [
    dict(
        type='MemoryMonitorHook',
        interval=50,
        log_system_memory=True,
    ),
    dict(
        type='GradientClipHook',
        max_norm=1.0,
        norm_type=2.0,
    ),
]"""


# 运行时配置
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn'),
)

work_dir = './work_dirs/dic_s_memory_optimized'
exp_name = 'dic_s_memory_optimized'