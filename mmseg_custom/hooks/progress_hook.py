"""
实时单行动态刷新式 ProgressHook - 修复版
- 修复字典格式化错误
- 正确处理 message_hub 返回值
- 每个 epoch 占用终端的一行
- 该行内容每 N iter 实时更新（使用 \r 回车符）
"""

import sys
import time
from typing import Optional, Dict, Any
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class ProgressHook(Hook):
    """每 epoch 一行实时更新的进度条 Hook"""
    
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    
    def __init__(
        self,
        interval: int = 50,
        print_epoch_summary: bool = True,
        progress_bar_width: int = 30,
    ):
        """
        Args:
            interval: 每 N 个 iter 更新一次终端
            print_epoch_summary: 是否打印 epoch 摘要
            progress_bar_width: 进度条宽度
        """
        self.interval = interval
        self.print_epoch_summary = print_epoch_summary
        self.progress_bar_width = progress_bar_width
        
        # 每个 epoch 的状态
        self.epoch_start_time = None
        self.epoch_losses = []
        self.best_miou = 0.0
        self.current_epoch = None
        self.dataloader_length = None
    
    def before_train_epoch(self, runner: Runner) -> None:
        """epoch 开始时初始化"""
        self.epoch_start_time = time.time()
        self.epoch_losses = []
        self.current_epoch = runner.epoch + 1
        
        # 获取数据加载器长度
        try:
            self.dataloader_length = len(runner.train_dataloader)
        except:
            try:
                self.dataloader_length = runner.train_loop.dataloader_length
            except:
                self.dataloader_length = 2975  # 默认 Cityscapes 值
    
    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: dict,
    ) -> None:
        """每次迭代后，每 N iter 更新一次终端"""
        
        # 记录 loss
        loss_value = self._safe_get_loss(runner)
        if loss_value is not None:
            self.epoch_losses.append(loss_value)
        
        # 判断是否需要更新终端（每 interval 个 iter）
        if batch_idx % self.interval != 0:
            return
        
        # 计算平均 loss
        avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else 0.0
        
        # 获取学习率
        lr_value = self._safe_get_lr(runner)
        
        # 获取 mIoU（如果有验证）
        miou_value = self._safe_get_miou(runner)
        miou_str = f"{miou_value:.4f}" if miou_value is not None else "----"
        
        # 计算进度百分比
        total_iters = self.dataloader_length
        progress_percent = (batch_idx + 1) / total_iters
        
        # 计算 ETA
        elapsed_time = time.time() - self.epoch_start_time
        if elapsed_time > 0 and progress_percent > 0:
            remaining_epochs = runner.max_epochs - self.current_epoch
            remaining_time_this_epoch = (elapsed_time / progress_percent) * (1 - progress_percent)
            time_per_epoch = elapsed_time / progress_percent
            total_eta_seconds = time_per_epoch * remaining_epochs + remaining_time_this_epoch
        else:
            total_eta_seconds = 0
        
        eta_str = self._format_time(total_eta_seconds)
        
        # 绘制进度条
        progress_bar = self._draw_progress_bar(progress_percent, self.progress_bar_width)
        
        # 🔑 关键：使用 \r 实时覆盖当前行（不产生新行）
        output_str = (
            f"\r[Epoch {self.current_epoch:3d}/{runner.max_epochs}] "
            f"[{batch_idx+1:5d}/{total_iters}] | "
            f"Loss: {avg_loss:7.4f} | "
            f"LR: {lr_value:.2e} | "
            f"mIoU: {miou_str:>6s} | "
            f"ETA: {eta_str:12s} | "
            f"{progress_bar}"
        )
        
        # 直接打印（绕过 log_level）
        print(output_str, end='', flush=True)
    
    def after_train_epoch(self, runner: Runner) -> None:
        """epoch 结束时，输出换行 + 摘要"""
        
        if not self.print_epoch_summary:
            return
        
        # 计算 epoch 统计
        epoch_time = time.time() - self.epoch_start_time
        avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else 0.0
        
        # 获取 mIoU
        val_miou = self._safe_get_miou(runner)
        
        # 检查是否是最佳模型
        is_best = False
        best_mark = ""
        if val_miou is not None and val_miou > self.best_miou:
            is_best = True
            self.best_miou = val_miou
            best_mark = " ⭐ BEST"
        
        # 获取学习率
        lr_value = self._safe_get_lr(runner)
        
        # 输出 epoch 摘要（使用 \n 换行，进入新行）
        epoch_summary = (
            f"\n[Epoch {self.current_epoch:3d}/{runner.max_epochs}] "
            f"Loss: {avg_loss:7.4f} | "
            f"LR: {lr_value:.2e} | "
            f"mIoU: {self._format_miou(val_miou):>6s}{best_mark} | "
            f"Time: {self._format_time(epoch_time)}"
        )
        
        print(epoch_summary)
        
        # 每 10 个 epoch 打印分隔线
        if self.current_epoch % 10 == 0:
            print("=" * 100)
    
    def _safe_get_loss(self, runner: Runner) -> Optional[float]:
        """安全地从 message_hub 获取 loss（修复字典问题）"""
        try:
            loss_dict = runner.message_hub.get_scalar('loss', 'current')
            
            # ✅ 处理所有可能的返回格式
            if isinstance(loss_dict, dict):
                # 情况 1：返回 {'current': value, ...}
                loss = loss_dict.get('current', None)
                if loss is not None:
                    return float(loss)
                # 情况 2：返回 {'loss_ce': value, ...}
                for key in loss_dict:
                    if 'loss' in key.lower():
                        val = loss_dict[key]
                        if isinstance(val, (int, float)):
                            return float(val)
                return None
            elif isinstance(loss_dict, (int, float)):
                # 直接返回数值
                return float(loss_dict)
            else:
                return None
        except Exception:
            return None
    
    def _safe_get_lr(self, runner: Runner) -> float:
        """安全地获取学习率"""
        try:
            lr = runner.optim_wrapper.get_lr()
            if isinstance(lr, (list, tuple)):
                return float(lr[0]) if lr else 0.0
            else:
                return float(lr)
        except Exception:
            return 0.0
    
    def _safe_get_miou(self, runner: Runner) -> Optional[float]:
        """安全地从 message_hub 获取 mIoU（修复字典问题）"""
        try:
            miou_dict = runner.message_hub.get_scalar('mIoU', 'current')
            
            # ✅ 处理所有可能的返回格式
            if isinstance(miou_dict, dict):
                # 情况 1：返回 {'current': value, ...}
                miou = miou_dict.get('current', None)
                if miou is not None:
                    return float(miou)
                # 情况 2：返回 {'mIoU': value, ...}
                miou = miou_dict.get('mIoU', None)
                if miou is not None:
                    return float(miou)
                # 情况 3：尝试获取任何数值字段
                for key, val in miou_dict.items():
                    if isinstance(val, (int, float)):
                        return float(val)
                return None
            elif isinstance(miou_dict, (int, float)):
                # 直接返回数值
                return float(miou_dict)
            else:
                return None
        except Exception:
            return None
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 0:
            return "N/A"
        
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def _format_miou(self, miou: Optional[float]) -> str:
        """格式化 mIoU"""
        return f"{miou:.4f}" if miou is not None else "----"
    
    def _draw_progress_bar(self, percent: float, width: int = 30) -> str:
        """绘制纯文本进度条"""
        percent = max(0, min(1.0, percent))
        filled = int(width * percent)
        
        bar = "=" * filled + "-" * (width - filled)
        return f"Progress: [{bar}] {percent*100:.1f}%"