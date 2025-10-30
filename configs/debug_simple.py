"""
简化调试配置 - 修复版本

专门用于测试数据加载问题，确保：
1. 正确导入所有必需的 transforms
2. 数据集返回正确的数据结构  
3. inputs 能正确转换为 tensor
"""

# ✅ 关键：导入自定义模块
custom_imports = dict(
    imports=['mmseg_custom','mmengine.dataset'],  # 确保 PackSegInputs 被导入
    allow_failed_imports=False
)

# 数据配置
data_root = '/root/projects/mmseg/datasets/cityscapes'

# ✅ 修复：最简化的 pipeline，确保每个 transform 都能正常工作
train_pipeline = [
    dict(type='LoadImageFromFile'),                                    # 1. 加载图像
    dict(type='LoadAnnotations'),                                      # 2. 加载分割标签
    dict(type='Resize', scale=(512, 512), keep_ratio=False),          # 3. Resize
    dict(type='PackSegInputs'),                                        # 4. 打包为最终格式
]

# 数据加载器配置
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,  # ✅ 关键：避免多进程问题
    sampler=dict(type='DefaultSampler', shuffle=False),

    dataset=dict(
        type='CityscapesACDCSimple',  # ✅ 使用修复后的数据集
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        pipeline=train_pipeline,
    ),
    #collate_fn=dict(type='pseudo_collate')
)

# 基本设置
default_scope = 'mmseg'
work_dir = './work_dirs/debug_simple'