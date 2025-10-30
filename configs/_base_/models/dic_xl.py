"""
DiC-XL 模型配置（XLarge）

参数：
- num_blocks: [10, 10, 8, 10, 10]
- channels: [160, 320, 640, 1280, 1280]
- 总参数量: ~140M
- 注意：需要 RTX 5090 或更高显存
"""

model = dict(
    type='DicSegmentor',
    arch='XL',
    num_classes=19,
    use_gating=True,
    use_condition=True,
    use_sparse_skip=True,
    num_weather_classes=5,
)