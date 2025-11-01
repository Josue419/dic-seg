"""
Cityscapes 数据集配置（晴天基准）

包含：
- train/val/test dataloader
- 统一的数据 pipeline
- 天气标签默认为 0 (clear)
"""



# 数据管道定义
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1024, 2048), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 1024), keep_ratio=False),
    dict(type='PackSegInputs'),
]

# 数据根目录
data_root = '/root/projects/mmseg/datasets/cityscapes'

# 训练数据加载
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CityscapesACDCDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        pipeline=train_pipeline,
    )
)

# 验证数据加载
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CityscapesACDCDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'
        ),
        pipeline=val_pipeline,
    )
)

# 测试数据加载
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CityscapesACDCDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'
        ),
        pipeline=val_pipeline,
    )
)

# 评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='val')
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='test')