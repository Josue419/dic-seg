"""
测试脚本 - 评估和推理

Usage:
    python tools/test.py configs/dic_s_cityscapes.py work_dirs/exp/latest.pth
"""

import argparse
import sys
from pathlib import Path

# ✅ 关键：导入 mmseg 以注册 transforms
import mmseg.datasets
import mmseg.models

from mmengine.config import Config
from mmengine.runner import Runner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description='Test DiC segmentation model')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--work-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = 'work_dirs/test'
    
    if args.seed is not None:
        cfg.seed = args.seed
    
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint, map_location='cpu')
    runner.test()


if __name__ == '__main__':
    main()