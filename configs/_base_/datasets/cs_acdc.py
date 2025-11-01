"""
Cityscapes + ACDC 联合训练数据集配置

包含：
- 训练：Cityscapes train + ACDC 所有天气 train
- 验证：Cityscapes val + ACDC 所有天气 val  
- 自动天气标签推断
- 数据平衡策略
"""



# 数据管道定义
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1024, 2048), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 1024), keep_ratio=False),
    dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
    dict(type='PackSegInputs'),
]

# 数据根目录
data_root = '/root/projects/mmseg/datasets'

# 联合训练数据加载器
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # Cityscapes train (clear weather)
            dict(
                type='CityscapesACDCDataset',
                data_root=f'{data_root}/cityscapes',
                data_prefix=dict(
                    img_path='leftImg8bit/train',
                    seg_map_path='gtFine/train'
                ),
                pipeline=train_pipeline,
            ),
            # ACDC fog train
            dict(
                type='CityscapesACDCDataset',
                data_root=f'{data_root}/acdc',
                data_prefix=dict(
                    img_path='rgb_anon/fog/train',
                    seg_map_path='gt/fog/train'
                ),
                pipeline=train_pipeline,
            ),
            # ACDC night train
            dict(
                type='CityscapesACDCDataset',
                data_root=f'{data_root}/acdc',
                data_prefix=dict(
                    img_path='rgb_anon/night/train',
                    seg_map_path='gt/night/train'
                ),
                pipeline=train_pipeline,
            ),
            # ACDC rain train
            dict(
                type='CityscapesACDCDataset',
                data_root=f'{data_root}/acdc',
                data_prefix=dict(
                    img_path='rgb_anon/rain/train',
                    seg_map_path='gt/rain/train'
                ),
                pipeline=train_pipeline,
            ),
            # ACDC snow train
            dict(
                type='CityscapesACDCDataset',
                data_root=f'{data_root}/acdc',
                data_prefix=dict(
                    img_path='rgb_anon/snow/train',
                    seg_map_path='gt/snow/train'
                ),
                pipeline=train_pipeline,
            ),
        ]
    )
)

# 联合验证数据加载器
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # Cityscapes val (clear weather)
            dict(
                type='CityscapesACDCDataset',
                data_root=f'{data_root}/cityscapes',
                data_prefix=dict(
                    img_path='leftImg8bit/val',
                    seg_map_path='gtFine/val'
                ),
                pipeline=val_pipeline,
            ),
            # ACDC fog val
            dict(
                type='CityscapesACDCDataset',
                data_root=f'{data_root}/acdc',
                data_prefix=dict(
                    img_path='rgb_anon/fog/val',
                    seg_map_path='gt/fog/val'
                ),
                pipeline=val_pipeline,
            ),
            # ACDC night val
            dict(
                type='CityscapesACDCDataset',
                data_root=f'{data_root}/acdc',
                data_prefix=dict(
                    img_path='rgb_anon/night/val',
                    seg_map_path='gt/night/val'
                ),
                pipeline=val_pipeline,
            ),
            # ACDC rain val
            dict(
                type='CityscapesACDCDataset',
                data_root=f'{data_root}/acdc',
                data_prefix=dict(
                    img_path='rgb_anon/rain/val',
                    seg_map_path='gt/rain/val'
                ),
                pipeline=val_pipeline,
            ),
            # ACDC snow val
            dict(
                type='CityscapesACDCDataset',
                data_root=f'{data_root}/acdc',
                data_prefix=dict(
                    img_path='rgb_anon/snow/val',
                    seg_map_path='gt/snow/val'
                ),
                pipeline=val_pipeline,
            ),
        ]
    )
)

# 测试数据加载器（使用验证集）
test_dataloader = val_dataloader

# 评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='val')
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='test')