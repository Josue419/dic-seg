import sys
import time
from typing import Optional, Dict, List
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.registry import HOOKS


@HOOKS.register_module()
class ProgressHook(Hook):
    """训练/验证双模式进度条 Hook"""

    def __init__(
        self,
        interval: int = 50,
        print_epoch_summary: bool = True,  # 保留参数，但内部逻辑已移除 epoch summary 打印
        progress_bar_width: int = 40,
    ):
        self.interval = interval
        # self.print_epoch_summary = print_epoch_summary # 不再使用此参数打印 summary
        self.progress_bar_width = progress_bar_width

        # 训练状态
        self.epoch_start_time = None
        self.epoch_losses = []
        self.current_epoch = None
        self.best_miou = 0.0
        self.current_miou = None

        # 验证状态
        self.val_start_time = None
        self.is_validating = False

    # ==================== Training ====================

    def before_train_epoch(self, runner: Runner) -> None:
        self.epoch_start_time = time.time()
        self.epoch_losses = []
        self.current_epoch = runner.epoch + 1

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: dict,
    ) -> None:
        loss_value = self._extract_loss_from_outputs(outputs)
        if loss_value is not None:
            self.epoch_losses.append(loss_value)

        if batch_idx % self.interval != 0 and batch_idx != len(runner.train_dataloader) - 1:
            return

        total_iters = len(runner.train_dataloader)
        current_iter = batch_idx + 1
        progress_percent = min(1.0, current_iter / total_iters)

        avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else 0.0
        lr_value = self._safe_get_current_lr(runner)

        elapsed_time = time.time() - self.epoch_start_time

        # --- 修正后的 ETA 计算 ---
        if elapsed_time > 0 and progress_percent > 0:
            # 计算当前 epoch 的预估总时间
            estimated_epoch_time = elapsed_time / progress_percent
            # 计算当前 epoch 的剩余时间
            remaining_time_this_epoch = estimated_epoch_time - elapsed_time

            # 估算每个 epoch 的平均时间（基于当前 epoch 的预估）
            # 注意：这仍然是一个粗糙的估算，因为每个 epoch 的时间可能不同
            # 更精确的方法需要记录历史 epoch 时间
            # 这里用当前 epoch 的预估时间作为基准
            estimated_time_per_epoch = estimated_epoch_time

            # 计算后续 epochs 的预估总剩余时间
            remaining_epochs = runner.max_epochs - self.current_epoch
            # 粗略估算：后续 epoch 每个都和当前 epoch 耗时相同
            remaining_time_future_epochs = estimated_time_per_epoch * remaining_epochs

            # 总的剩余时间 = 当前 epoch 剩余 + 后续 epochs 剩余
            total_remaining_time = remaining_time_this_epoch + remaining_time_future_epochs
        else:
            # 在训练刚开始时，无法估算
            total_remaining_time = 0
            estimated_epoch_time = 0 # 用于显示已用/预估总时长的场景

        # --- 选择显示格式 ---
        # 1. 显示 已用时长 / 预估总时长 (原逻辑，但计算更清晰)
        # elapsed_str = self._format_time(elapsed_time)
        # total_estimated_str = self._format_time(estimated_epoch_time + estimated_time_per_epoch * (runner.max_epochs - self.current_epoch - 1))
        # eta_str = f"{elapsed_str}/{total_estimated_str}"

        # 2. 显示 已用时长 / 动态剩余时长 (推荐，符合直觉)
        elapsed_str = self._format_time(elapsed_time)
        eta_str = self._format_time(total_remaining_time)

        progress_bar = self._draw_progress_bar(progress_percent, self.progress_bar_width)

        output_str = (
            f"\r[Train Epoch {self.current_epoch:3d}/{runner.max_epochs}] "
            f"[{current_iter:5d}/{total_iters}] | "
            f"Loss: {avg_loss:7.4f} | "
            f"LR: {lr_value:.2e} | "
            f"Time/ETA: {elapsed_str}/{eta_str} | " # 显示 已用/剩余
            f"{progress_bar}"
        )
        print(output_str, end='', flush=True)

    def after_train_epoch(self, runner: Runner) -> None:
        print()  # 换行，结束当前 epoch 的动态行
        # ✅ 移除 epoch summary 打印逻辑
        # if not self.print_epoch_summary:
        #     return
        # ... 其余 summary 逻辑已删除 ...

    # ==================== Validation ====================

    def before_val_epoch(self, runner: Runner) -> None:
        # ✅ 移除 "Starting..." 提示
        self.val_start_time = time.time()
        self.is_validating = True
        # print(f"\n[Validation Epoch {self.current_epoch}] Starting...")

    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: dict,
    ) -> None:
        if not self.is_validating:
            return

        total_iters = len(runner.val_dataloader)
        current_iter = batch_idx + 1
        if current_iter % self.interval != 0 and current_iter != total_iters:
            return

        progress_percent = min(1.0, current_iter / total_iters)
        elapsed = time.time() - self.val_start_time
        speed = elapsed / current_iter if current_iter > 0 else 0
        eta = speed * (total_iters - current_iter)

        progress_bar = self._draw_progress_bar(progress_percent, self.progress_bar_width)
        eta_str = self._format_time(eta)

        output_str = (
            f"\r[Val Epoch {self.current_epoch}] "
            f"[{current_iter:4d}/{total_iters}] | "
            f"ETA: {eta_str:8s} | "
            f"{progress_bar}"
        )
        print(output_str, end='', flush=True)

    def after_val_epoch(
        self, 
        runner: Runner, 
        metrics: Dict[str, float] = None  # ✅ 关键修复：添加 metrics 参数
    ) -> None:
        """验证完成后调用
        
        Args:
            runner: MMEngine Runner
            metrics: 验证指标字典，由 MMEngine 自动传入
        """
        self.is_validating = False
        print()  # 换行

        # ✅ 关键修复：优先从 metrics 参数中获取 mIoU
        # 请确保此处的 key 与你的配置文件中 evaluation.save_best 的值一致
        # 例如，如果 save_best='val/mIoU'，则使用 'val/mIoU'
        miou_key = 'val/mIoU' # 请根据你的实际配置修改这个键名
        if metrics is not None and isinstance(metrics, dict) and miou_key in metrics:
            try:
                self.current_miou = float(metrics[miou_key])
            except (ValueError, TypeError):
                self.current_miou = None
        else:
            # 备选：从 message_hub 读取（通常在验证后才有值，且可能带有前缀）
            try:
                # 尝试直接读取带前缀的 key
                miou_val = runner.message_hub.get_scalar(miou_key, 'current')
                if isinstance(miou_val, dict):
                    miou_val = miou_val.get('current')
                if isinstance(miou_val, (int, float)):
                    self.current_miou = float(miou_val)
            except Exception:
                self.current_miou = None

        val_time = time.time() - self.val_start_time
        miou_str = f"{self.current_miou:.4f}" if self.current_miou is not None else "----"

        # 打印验证结果
        val_summary = (
            f"[Validation Epoch {self.current_epoch}] "
            f"mIoU: {miou_str} | "
            f"Time: {self._format_time(val_time)}"
        )
        print(val_summary)

        # 更新最佳
        if self.current_miou is not None and self.current_miou > self.best_miou:
            self.best_miou = self.current_miou
            print(f"⭐ New Best mIoU: {self.best_miou:.4f}")

    # ==================== Utils ====================

    def _extract_loss_from_outputs(self, outputs: dict) -> Optional[float]:
        try:
            if 'loss' in outputs:
                loss = outputs['loss']
                if isinstance(loss, dict):
                    # 如果 loss 是一个字典，尝试获取 'loss' 键或其他可能的键
                    if 'loss' in loss:
                        loss = loss['loss']
                    else:
                        # 如果字典中还有其他可能代表 loss 的键，可以在这里处理
                        # 例如，如果 loss 是 {'main_loss': ..., 'aux_loss': ...} 的形式
                        # 这里简单地取第一个值
                        loss = next(iter(loss.values()))
                return float(loss)
            # 有时 loss 可能直接是 tensor 或数字
            if isinstance(outputs, (float, int)):
                return float(outputs)
        except (TypeError, ValueError, AttributeError):
            pass
        return None

    def _safe_get_current_lr(self, runner: Runner) -> float:
        try:
            # 尝试方法 1: 使用 optim_wrapper 的 get_lr 方法 (MMEngine 推荐)
            if hasattr(runner, 'optim_wrapper') and runner.optim_wrapper is not None:
                lr_list = runner.optim_wrapper.get_lr()
                if isinstance(lr_list, (list, tuple)) and len(lr_list) > 0:
                    # get_lr 可能返回一个列表，取第一个参数组的学习率
                    return float(lr_list[0])
                elif isinstance(lr_list, (int, float)):
                    # get_lr 也可能返回单个值
                    return float(lr_list)
                # 如果 get_lr 返回了空列表或无效类型，继续尝试方法 2
        except (AttributeError, IndexError, TypeError, ValueError) as e:
            # print(f"DEBUG: get_lr failed: {e}") # 可选：调试用
            pass

        try:
            # 尝试方法 2: 直接访问优化器参数组
            if (hasattr(runner, 'optim_wrapper') and
                runner.optim_wrapper is not None and
                hasattr(runner.optim_wrapper, 'optimizer')):
                optimizer = runner.optim_wrapper.optimizer
                if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                    # 取第一个参数组的学习率
                    lr = optimizer.param_groups[0].get('lr', 0.0)
                    return float(lr)
        except (AttributeError, IndexError, TypeError, ValueError) as e:
            # print(f"DEBUG: Direct access failed: {e}") # 可选：调试用
            pass

        # 如果所有方法都失败，返回 0.0
        return 0.0

    def _format_time(self, seconds: float) -> str:
        if seconds < 0:
            return "N/A"
        # 确保至少显示一个单位，即使为 0
        seconds = int(seconds)
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        # 构建非零部分的列表
        parts = []
        if days > 0:
            parts.append(f"{days}d")
            if hours > 0:
                parts.append(f"{hours}h")
                # 如果天数大于0，通常小时数也显示，即使分钟为0
                # 但这里遵循只显示前两个非零单位的逻辑，除非天数和小时数都为0
            elif minutes > 0: # 天数>0, 小时=0, 分钟>0
                parts.append(f"{minutes}m")
            elif secs > 0: # 天数>0, 小时=0, 分钟=0, 秒>0
                 parts.append(f"{secs}s")
            else: # 全为0
                parts.append(f"{days}d") # 如果只有天数是0，但其他都为0，至少显示一个
        elif hours > 0:
            parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            elif secs > 0:
                parts.append(f"{secs}s")
        elif minutes > 0:
            parts.append(f"{minutes}m")
            if secs > 0:
                parts.append(f"{secs}s")
        elif secs > 0:
            parts.append(f"{secs}s")
        else:
            # 如果所有时间单位都是0
            return "0s"
        
        # 只取前两个非零部分
        return " ".join(parts[:2]) if parts else "0s"


    def _draw_progress_bar(self, percent: float, width: int = 40) -> str:
        percent = max(0.0, min(1.0, percent))
        filled = int(width * percent)
        bar = "=" * filled + "-" * (width - filled)
        return f"[{bar}] {percent*100:.1f}%"
