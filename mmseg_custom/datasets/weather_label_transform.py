"""
自定义 Transform 用于处理天气标签
"""

from typing import Dict
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class LoadWeatherLabel(BaseTransform):
    """加载天气标签到 data_samples.metainfo"""
    
    def transform(self, results: Dict) -> Dict:
        """
        Args:
            results (dict): 包含 weather_label 的结果字典
        
        Returns:
            dict: 更新后的结果字典
        """
        weather_label = results.get('weather_label', 0)
        
        # 确保 data_samples 存在
        if 'data_samples' not in results:
            # 如果还没有 data_samples，先存储到临时位置
            results['weather_label_temp'] = weather_label
        else:
            # 如果已经有 data_samples，直接添加到 metainfo
            if not hasattr(results['data_samples'], 'metainfo'):
                results['data_samples'].metainfo = {}
            results['data_samples'].metainfo['weather_label'] = weather_label
        
        return results


@TRANSFORMS.register_module()
class FinalizeWeatherLabel(BaseTransform):
    """最终处理天气标签（在 PackSegInputs 之后）"""
    
    def transform(self, results: Dict) -> Dict:
        """
        Args:
            results (dict): PackSegInputs 处理后的结果
        
        Returns:
            dict: 添加天气标签后的结果
        """
        # 如果有临时存储的天气标签，现在添加到 data_samples
        if 'weather_label_temp' in results and 'data_samples' in results:
            weather_label = results.pop('weather_label_temp')
            
            if not hasattr(results['data_samples'], 'metainfo'):
                results['data_samples'].metainfo = {}
            results['data_samples'].metainfo['weather_label'] = weather_label
        
        return results