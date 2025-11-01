"""
DiC-S 显存优化调试配置（RTX 4090）
"""

_base_ = [
    '_base_/models/dic_s.py',
    '_base_/schedules/poly_50e.py',
    '_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg_custom','mmengine.dataset'],  
    allow_failed_imports=False
)

# ✅ 显存优化的数据管道
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),  # 固定尺寸，减少显存波动
    dict(type='RandomFlip', prob=0.5),
    # ✅ 正确的归一化参数
    dict(type='Normalize', 
         mean=[123.675, 116.28, 103.53], 
         std=[58.395, 57.12, 57.375], 
         to_rgb=True),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False), 
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', 
         mean=[123.675, 116.28, 103.53], 
         std=[58.395, 57.12, 57.375], 
         to_rgb=True),
    dict(type='PackSegInputs'),
]

# ✅ 显存优化的数据加载
data_root = '/root/projects/mmseg/datasets'

train_dataloader = dict(
    batch_size=1,
    num_workers=0,  # 避免多进程显存问题
    persistent_workers=False,
    pin_memory=False,  # 关闭 pin_memory 减少显存占用
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CityscapesACDCDataset',
        data_root=f'{data_root}/acdc',
        data_prefix=dict(
            img_path='rgb_anon/fog/train',  # 只用一个天气类型调试
            seg_map_path='gt/fog/train'
        ),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CityscapesACDCDataset', 
        data_root=f'{data_root}/acdc',
        data_prefix=dict(
            img_path='rgb_anon/fog/val',
            seg_map_path='gt/fog/val'
        ),
        pipeline=val_pipeline,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='val')
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='test')

# ✅ 显存优化的模型配置
model = dict(
    type='DicSegmentor',
    arch='S',
    num_classes=19,
    use_gating=True,
    use_condition=True,
    use_sparse_skip=True,
    num_weather_classes=5,
)

# ✅ 启用混合精度训练
optim_wrapper = dict(
    type='AmpOptimWrapper',  # 自动混合精度
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=1.0),
)

# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=0, end=2, by_epoch=True)
]

# 运行时设置
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn'),  # 多进程优化
)

work_dir = './work_dirs/dic_s_debug_memory'