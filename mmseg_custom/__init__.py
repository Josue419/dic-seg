"""MMSeg Custom Models Package.

关键：确保优先使用 mmseg 的 transforms 而不是 mmcv 的
同时修复 pseudo_collate 的使用问题
"""

# ✅ 第一步：优先导入 mmseg 所有模块
import mmseg
import mmseg.datasets
import mmseg.models
import mmseg.datasets.transforms
import mmseg.datasets.transforms.loading

# ✅ 第二步：确保 mmseg transforms 覆盖 mmcv transforms
try:
    from mmseg.registry import TRANSFORMS as MMSEG_TRANSFORMS
    from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS

    # 将 mmseg 的 transforms 强制注册到 mmengine registry，覆盖 mmcv 版本
    mmseg_transform_names = list(MMSEG_TRANSFORMS._module_dict.keys())
    for name in mmseg_transform_names:
        transform_cls = MMSEG_TRANSFORMS.get(name)
        if transform_cls is not None:
            # 强制覆盖，确保 mmengine 使用 mmseg 版本
            MMENGINE_TRANSFORMS.register_module(name=name, module=transform_cls, force=True)

    print(f"✅ 成功注册 {len(mmseg_transform_names)} 个 mmseg transforms 到 mmengine registry")
    
    # 验证关键 transforms
    key_transforms = ['LoadAnnotations', 'PackSegInputs', 'LoadImageFromFile']
    for name in key_transforms:
        cls = MMENGINE_TRANSFORMS.get(name)
        if cls and 'mmseg' in str(cls):
            print(f"  ✅ {name}: 使用 mmseg 版本")
        elif cls:
            print(f"  ⚠ {name}: 使用 {cls}")
        else:
            print(f"  ❌ {name}: 未找到")
            
except Exception as e:
    print(f"⚠ Transform 注册表修复失败: {e}")

# ✅ 第三步：导入自定义模块
from .models import *
from .datasets import *

__version__ = '1.0.0'

__all__ = [
    'DicEncoder',
    'DicDecoder', 
    'DicSegmentor',
    'CityscapesACDCDataset',
    'CityscapesACDCSimple',
    'LoadWeatherLabel', 
    'FinalizeWeatherLabel'
]