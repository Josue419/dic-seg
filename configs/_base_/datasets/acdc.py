"""
ACDC 数据集配置（恶劣天气鲁棒性）

包含：
- 4 种天气类型（fog, night, rain, snow）
- 每种天气独立数据加载
"""
custom_imports = dict(
    imports=[
        'mmseg.datasets.transforms.formatting',  # 用于注册 PackSegInputs
        'mmseg_custom.models'                    # 用于注册你的自定义模型
    ],
    allow_failed_imports=False
)
# 数据管道定义（与 Cityscapes 相同）
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1024, 2048),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 1024), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

# 训练数据加载（只用 ACDC，无 Cityscapes）
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # ACDC fog
            dict(
                type='CityscapesACDCDataset',
                data_root='../mmseg/datasets/acdc',
                data_prefix=dict(img_path='rgb_anon/fog/train', seg_map_path='gt/fog/train'),
                pipeline=train_pipeline,
            ),
            # ACDC night
            dict(
                type='CityscapesACDCDataset',
                data_root='../mmseg/datasets/acdc',
                data_prefix=dict(img_path='rgb_anon/night/train', seg_map_path='gt/night/train'),
                pipeline=train_pipeline,
            ),
            # ACDC rain
            dict(
                type='CityscapesACDCDataset',
                data_root='../mmseg/datasets/acdc',
                data_prefix=dict(img_path='rgb_anon/rain/train', seg_map_path='gt/rain/train'),
                pipeline=train_pipeline,
            ),
            # ACDC snow
            dict(
                type='CityscapesACDCDataset',
                data_root='../mmseg/datasets/acdc',
                data_prefix=dict(img_path='rgb_anon/snow/train', seg_map_path='gt/snow/train'),
                pipeline=train_pipeline,
            ),
        ]
    )
)

# 验证数据加载（用 Cityscapes）
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesACDCDataset',
        data_root='../mmseg/datasets/cityscapes',
        data_prefix=dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=val_pipeline,
    )
)

# 测试数据加载
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesACDCDataset',
        data_root='../mmseg/datasets/cityscapes',
        data_prefix=dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=val_pipeline,
    )
)

# 评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])