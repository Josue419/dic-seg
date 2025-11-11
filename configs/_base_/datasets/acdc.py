"""
ACDC 数据集配置 - 修复版

关键修复：
1. 正确的数据归一化参数
2. 使用预处理标签
3. 统一的 data_root 路径
"""

# 数据管道定义 - 修复版
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(128,256), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(128,128), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # 修复：使用正确的 ImageNet 归一化参数
    dict(type='Normalize', 
         mean=[123.675, 116.28, 103.53], 
         std=[58.395, 57.12, 57.375], 
         to_rgb=True),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(128,256), keep_ratio=False),
    # 修复：验证时也需要归一化
    dict(type='Normalize', 
         mean=[123.675, 116.28, 103.53], 
         std=[58.395, 57.12, 57.375], 
         to_rgb=True),
    dict(type='PackSegInputs'),
]

# 修复：统一的数据根目录
data_root = '/root/projects/mmseg/datasets/acdc'

# 训练数据加载（ACDC 所有天气类型）
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # ACDC fog
            dict(
                type='CityscapesACDCDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='rgb_anon/fog/train',
                    seg_map_path='gt/fog/train'
                ),
                pipeline=train_pipeline,
                use_processed_labels=True,  # 修复：启用预处理标签
            ),
            # ACDC night
            dict(
                type='CityscapesACDCDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='rgb_anon/night/train',
                    seg_map_path='gt/night/train'
                ),
                pipeline=train_pipeline,
                use_processed_labels=True,
            ),
            # ACDC rain
            dict(
                type='CityscapesACDCDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='rgb_anon/rain/train',
                    seg_map_path='gt/rain/train'
                ),
                pipeline=train_pipeline,
                use_processed_labels=True,
            ),
            # ACDC snow
            dict(
                type='CityscapesACDCDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='rgb_anon/snow/train',
                    seg_map_path='gt/snow/train'
                ),
                pipeline=train_pipeline,
                use_processed_labels=True,
            ),
        ]
    )
)

# 验证数据加载（ACDC 所有天气类型的验证集）
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # ACDC fog val
            dict(
                type='CityscapesACDCDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='rgb_anon/fog/val',
                    seg_map_path='gt/fog/val'
                ),
                pipeline=val_pipeline,
                use_processed_labels=True,
            ),
            # ACDC night val
            dict(
                type='CityscapesACDCDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='rgb_anon/night/val',
                    seg_map_path='gt/night/val'
                ),
                pipeline=val_pipeline,
                use_processed_labels=True,
            ),
            # ACDC rain val
            dict(
                type='CityscapesACDCDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='rgb_anon/rain/val',
                    seg_map_path='gt/rain/val'
                ),
                pipeline=val_pipeline,
                use_processed_labels=True,
            ),
            # ACDC snow val
            dict(
                type='CityscapesACDCDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='rgb_anon/snow/val',
                    seg_map_path='gt/snow/val'
                ),
                pipeline=val_pipeline,
                use_processed_labels=True,
            ),
        ]
    )
)

# 测试数据加载（使用验证集）
test_dataloader = val_dataloader

# 评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='val')
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='test')