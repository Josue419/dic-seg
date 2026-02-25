_base_ = [
    '../_base_/models/segmamba_ohem.py',
    '../_base_/datasets/bdd100k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]


crop_size = (1280, 1280)
data_preprocessor = dict(size = crop_size)

# 若没有可用的 Mamba 预训练权重,init_cfg 设置为 None；如果有请替换 checkpoint 链接
# 例如: init_cfg=dict(type='Pretrained', checkpoint='path/to/mamba_pretrain.pth')
checkpoint = None

model = dict(
    
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,   # 与 init_cfg 不能并用；保持与 mmseg 风格一致
    backbone=dict(

        init_cfg=dict(type='Pretrained', checkpoint=checkpoint) if checkpoint else None
    ),
    test_cfg=dict(
        mode='whole')
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
# Dataloaders (可根据显存调整 batch_size)
train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
#load_from = '/root/projects/seg/work_dirs/segmamba_b0-cs/iter_20000.pth'
