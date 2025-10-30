"""
训练脚本 - 支持 MMSeg v1.2.2 + MMEngine 0.9.x

Usage:
    python tools/train.py configs/dic_s_cityscapes_debug.py
    python tools/train.py configs/dic_s_cityscapes.py --work-dir work_dirs/custom
"""

import argparse
import sys
from pathlib import Path

# ✅ 关键：导入 mmseg 以注册 transforms
import mmseg.datasets
import mmseg.models

from mmengine.config import Config
from mmengine.runner import Runner

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description='Train DiC segmentation model')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--work-dir', type=str, default=None)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--load-from', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 更新配置
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    if args.resume_from is not None:
        cfg.resume = True
        cfg.load_from = None
        cfg.resume_from = args.resume_from
    
    if args.load_from is not None:
        cfg.load_from = args.load_from
    
    if args.seed is not None:
        cfg.seed = args.seed
    
    if args.deterministic:
        cfg.deterministic = True
    
    # 构建并运行
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()