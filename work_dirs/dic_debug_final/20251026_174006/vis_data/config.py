custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmseg_custom',
    ])
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_best='mIoU',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmseg'
deterministic = True
env_cfg = dict(cudnn_benchmark=True)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')
exp_name = 'dic_s_cityscapes_debug'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    arch='S',
    num_classes=19,
    num_weather_classes=5,
    type='DicSegmentor',
    use_condition=True,
    use_gating=True,
    use_sparse_skip=True)
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
        end=50,
        eta_min=1e-06,
        power=0.9,
        type='PolyLR'),
]
resume = False
resume_from = None
seed = 2025
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root='../mmseg/datasets/cityscapes',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesACDCDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
train_cfg = dict(max_epochs=2, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root='../mmseg/datasets/cityscapes',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    2048,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesACDCDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1024,
            2048,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root='../mmseg/datasets/cityscapes',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesACDCDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        512,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
work_dir = 'work_dirs/dic_debug_final'
