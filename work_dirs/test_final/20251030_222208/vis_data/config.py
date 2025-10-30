custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmseg_custom',
        'mmengine.dataset',
    ])
data_root = '/root/projects/mmseg/datasets/cityscapes'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmseg'
deterministic = True
env_cfg = dict(cudnn_benchmark=True)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    arch='S',
    num_classes=19,
    num_weather_classes=5,
    type='DicSegmentor',
    use_condition=False,
    use_gating=False,
    use_sparse_skip=False)
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), lr=6e-05, type='AdamW', weight_decay=0.01)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=200,
        eta_min=1e-06,
        power=0.9,
        type='PolyLR'),
]
seed = 2025
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root='/root/projects/mmseg/datasets/cityscapes',
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesACDCSimple'),
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], prefix='test', type='IoUMetric')
train_cfg = dict(max_epochs=2, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root='/root/projects/mmseg/datasets/cityscapes',
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesACDCSimple'),
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(to_float32=True, type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root='/root/projects/mmseg/datasets/cityscapes',
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesACDCSimple'),
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], prefix='val', type='IoUMetric')
val_pipeline = [
    dict(to_float32=True, type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='PackSegInputs'),
]
work_dir = 'work_dirs/test_final'
