"""
Debug: 2 epochs on Cityscapes - 最终修复版

关键修复：
1. 正确使用 pseudo_collate 函数
2. 确保 Transform 注册表问题解决
3. 完善数据加载器配置
"""
#from mmengine.dataset import default_collate
# ============================================================================
# 第一步：导入自定义模块
# ============================================================================

custom_imports = dict(
    imports=['mmseg_custom','mmengine.dataset'],  
    allow_failed_imports=False
)

# ============================================================================
# 第二步：模型配置
# ============================================================================

model = dict(
    type='DicSegmentor',
    arch='S',
    num_classes=19,
    use_gating=True,
    use_condition=False,
    use_sparse_skip=False,
    num_weather_classes=5,
)

# ============================================================================
# 第三步：数据配置
# ============================================================================

data_root = '/root/projects/mmseg/datasets/cityscapes'

# Pipeline 配置
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CityscapesACDCSimple',
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CityscapesACDCSimple',
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'
        ),
        pipeline=val_pipeline,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'],prefix='val')
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'],prefix='test')

# ============================================================================
# 第四步：优化器
# ============================================================================

optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None
)

# ============================================================================
# 第五步：学习率调度
# ============================================================================

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=200,
        by_epoch=True,
    )
]

# ============================================================================
# 第六步：训练循环
# ============================================================================

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# 第七步：运行时配置
# ============================================================================

default_scope = 'mmseg'
env_cfg = dict(cudnn_benchmark=True)
log_processor = dict(by_epoch=True)
log_level = 'INFO'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best='mIoU',
    ),
)

evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')

work_dir = './work_dirs/dic_debug_final'