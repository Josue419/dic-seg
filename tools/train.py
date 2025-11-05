"""
精简高效的训练脚本 - 保留日志重定向、系统信息、配置摘要、美观输出
"""

import argparse
import sys
import os
import logging
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]  # 假设 tools/train.py，上两级是项目根
sys.path.insert(0, str(project_root))
import psutil
import torch

# ============================================================================
# 日志重定向：必须在 mmengine 导入后立即设置
# ============================================================================

def redirect_mmengine_logs(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    for name in ['mmengine', 'mmseg', 'mmcv']:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = False
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


# 导入必要模块
import mmseg.datasets  # noqa
import mmseg.models    # noqa
from mmengine.config import Config
from mmengine.runner import Runner


# 自定义颜色（保留关键视觉提示）
class C:
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


def print_section(title: str, char='-', width=80):
    print(f"\n{C.BOLD}{C.BLUE}{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}{C.ENDC}")


def print_system_info():
    info = {
        'Python': f"{sys.version.split()[0]}",
        'PyTorch': torch.__version__,
        'CUDA': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'cuDNN': torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A',
        'CPU Cores': psutil.cpu_count(logical=False),
        'RAM (GB)': f"{psutil.virtual_memory().available / 1e9:.1f}/{psutil.virtual_memory().total / 1e9:.1f}",
    }
    if torch.cuda.is_available():
        gpus = [f"GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)"
                for i in range(torch.cuda.device_count())]
        info['GPUs'] = gpus

    print_section("🖥️  SYSTEM INFORMATION")
    for k, v in info.items():
        if k == 'GPUs':
            print(f"  {k}:")
            for gpu in v:
                print(f"    {gpu}")
        else:
            print(f"  {k}: {v}")


def print_config_summary(cfg):
    c = cfg
    print_section("⚙️  TRAINING CONFIGURATION")

    # Model
    m = c.model
    print(f"{C.YELLOW}[Model]{C.ENDC}")
    print(f"  Type: {m.type}, Arch: {m.get('arch', 'N/A')}, Classes: {m.num_classes}")
    print(f"  Gating: {m.get('use_gating', False)}, Condition: {m.get('use_condition', False)}")
    print(f"  Gradient Checkpointing: {m.get('gradient_checkpointing', False)}")

    # Data
    train_dl = c.train_dataloader
    val_dl = c.val_dataloader
    print(f"\n{C.YELLOW}[Data]{C.ENDC}")
    print(f"  Train: batch={train_dl.batch_size}, workers={train_dl.num_workers}")
    print(f"  Val:   batch={val_dl.batch_size}, workers={val_dl.num_workers}")

    # Training
    print(f"\n{C.YELLOW}[Training]{C.ENDC}")
    print(f"  Epochs: {c.train_cfg.max_epochs}")
    opt = c.optim_wrapper.optimizer
    print(f"  Optim: {opt.type}, LR: {opt.lr}, WD: {opt.weight_decay}")

    sched = c.param_scheduler[0] if isinstance(c.param_scheduler, list) else c.param_scheduler
    if sched.type == 'PolyLR':
        print(f"  LR Schedule: PolyLR (power={sched.power}, eta_min={sched.eta_min})")

    print(f"\n{C.YELLOW}[Output]{C.ENDC}")
    print(f"  Work Dir: {c.work_dir}")
    print(f"  Exp Name: {c.get('exp_name', 'default')}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--work-dir', default=None)
    parser.add_argument('--resume-from', default=None)
    parser.add_argument('--load-from', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.work_dir: cfg.work_dir = args.work_dir
    if args.resume_from:
        cfg.resume = True
        cfg.resume_from = args.resume_from
    if args.load_from:
        cfg.load_from = args.load_from
    if args.seed is not None:
        cfg.seed = args.seed
    if args.deterministic:
        cfg.deterministic = True

    # Setup logging
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    redirect_mmengine_logs(work_dir / 'mmengine_debug.log')

    # Suppress MMEngine's config print
    cfg.log_level = 'WARNING'

    # Print info
    print_section("DiC Semantic Segmentation Training", char='#', width=80)
    print_system_info()
    print_config_summary(cfg)

    print_section("🚀 STARTING TRAINING")
    print(f"{C.BOLD}Output dir:{C.ENDC} {cfg.work_dir}")

    # Train
    try:
        runner = Runner.from_cfg(cfg)
        runner.train()
        print(f"\n{C.BOLD}{C.GREEN}✅ Training completed!{C.ENDC}")
        print(f"Results: {cfg.work_dir}")
    except KeyboardInterrupt:
        print(f"\n{C.BOLD}{C.YELLOW}⚠️  Interrupted by user{C.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{C.BOLD}{C.RED}❌ Error: {e}{C.ENDC}")
        raise


if __name__ == '__main__':
    main()