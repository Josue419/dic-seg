"""
200 epoch PolyLR 学习率策略（完整训练）

使用：
- 优化器：AdamW
- 学习率调度：PolyLR (power=0.9)
- 训练循环：200 epochs
"""

# 优化器
optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
)

# 学习率调度：PolyLR
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

# 训练循环
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')