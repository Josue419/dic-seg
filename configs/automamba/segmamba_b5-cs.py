_base_ = ['./segmamba_b0-cs.py']

checkpoint = None  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))


'''
B1:
embed_dims=64
num_layers=[2, 2, 2, 2]
num_heads=[1, 2, 5, 8]
decode_head.in_channels=[64, 128, 320, 512]
B2:
embed_dims=64
num_layers=[3, 4, 6, 3]
num_heads=[1, 2, 5, 8]
decode_head.in_channels=[64, 128, 320, 512]
B3:
embed_dims=64
num_layers=[3, 4, 18, 3]
num_heads=[1, 2, 5, 8]
decode_head.in_channels=[64, 128, 320, 512]
B4:
embed_dims=64
num_layers=[3, 8, 27, 3]
num_heads=[1, 2, 5, 8]
decode_head.in_channels=[64, 128, 320, 512]
B5:
embed_dims=64
num_layers=[3, 6, 40, 3]
num_heads=[1, 2, 5, 8]
decode_head.in_channels=[64, 128, 320, 512]
'''