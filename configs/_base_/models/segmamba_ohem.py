# model settings
custom_imports = dict(imports=['mmseg.models.backbones.sm_dwconv_wh'], allow_failed_imports=False)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionMamba',  # switched to Mamba backbone
        with_dwconv=True,
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        d_state=16,
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # ------------------- 新增 OHEM 配置 -------------------
        sampler=dict(
            type='OHEMPixelSampler', 
            thresh=0.7,       # 只有预测概率低于 0.7 的像素才会被视为难样本参与 Loss 计算
            min_kept=100000   # 无论如何，每张图片至少保留 100,000 个像素参与计算（防止训练初期 Loss 为 0）
        )
        # -----------------------------------------------------
    ),
        auxiliary_head=dict(
        type='FCNHead',                   # 使用简单的 FCN Head
        in_channels=160,                  # Stage 3 的输出通道数 (确保你的 embed_dims=32, num_heads[2]=5)
        in_index=2,                       # 使用 Stage 3 的特征
        channels=256,                     # 【建议】改为 256 以获得更好的特征映射能力 (64 也可以)
        num_convs=1,                      # 卷积层数
        concat_input=False,
        dropout_ratio=0.1,                
        num_classes=19,
        norm_cfg=norm_cfg, 
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4               # 权重 0.4 是标准做法
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))