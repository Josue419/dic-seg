"""
50 epoch PolyLR 学习率策略（快速验证）

用于：
- 消融实验
- 快速原型验证
- 硬件测试
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

# 学习率调度
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=50,
        by_epoch=True,
    )
]

# 训练循环
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')