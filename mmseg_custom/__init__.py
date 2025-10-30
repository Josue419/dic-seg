"""MMSeg Custom Models Package.

关键：必须在此处导入所有自定义模块，以确保 @register_module() 装饰器被执行
"""

# ✅ 第一步：导入 mmseg 官方模块以注册其 transforms
import mmseg.datasets
import mmseg.models

# ✅ 第二步：导入自定义模块，触发 @MODELS.register_module() 装饰器
from .models import *
from .datasets import *

__version__ = '1.0.0'

__all__ = [
    'DicEncoder',
    'DicDecoder',
    'DicSegmentor',
    'CityscapesACDCDataset',
]