"""
DiC-B 模型配置（Base）

参数：
- num_blocks: [8, 8, 6, 8, 8]
- channels: [128, 256, 512, 1024, 1024]
- 总参数量: ~90M
"""

model = dict(
    type='DicSegmentor',
    arch='B',
    num_classes=19,
    use_gating=True,
    use_condition=True,
    use_sparse_skip=True,
    num_weather_classes=5,
)