"""
DiC-S 模型配置（Small）

参数：
- num_blocks: [6, 6, 5, 6, 6]
- channels: [96, 192, 384, 768, 768]
- 总参数量: ~50M
"""

model = dict(
    type='DicSegmentor',
    arch='S',
    num_classes=19,
    use_gating=True,
    use_condition=True,
    use_sparse_skip=True,
    num_weather_classes=5,
    gradient_checkpointing=True,  # ✅ 启用梯度检查点
)