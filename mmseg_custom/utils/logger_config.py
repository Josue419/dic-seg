"""
美化终端输出的日志配置工具
"""

import logging
from pathlib import Path
from typing import Optional
import sys

# 定义颜色（支持 Linux/Windows 10+ ANSI）
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class CompactFormatter(logging.Formatter):
    """紧凑型日志格式化器"""
    
    FORMATS = {
        logging.DEBUG: f"{Colors.CYAN}[DEBUG]{Colors.ENDC} %(message)s",
        logging.INFO: f"{Colors.GREEN}[INFO]{Colors.ENDC} %(message)s",
        logging.WARNING: f"{Colors.YELLOW}[WARN]{Colors.ENDC} %(message)s",
        logging.ERROR: f"{Colors.RED}[ERROR]{Colors.ENDC} %(message)s",
        logging.CRITICAL: f"{Colors.RED}{Colors.BOLD}[FATAL]{Colors.ENDC} %(message)s",
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """设置美化的日志记录器"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # 清除旧 handler
    logger.handlers.clear()
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CompactFormatter())
    logger.addHandler(console_handler)
    
    # 文件输出（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

class ConfigPrinter:
    """美化配置打印"""
    
    @staticmethod
    def print_config_summary(cfg, max_lines: int = 20):
        """打印简洁的配置摘要（而非完整 config）"""
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
        print(f"⚙️  TRAINING CONFIGURATION SUMMARY")
        print(f"{'='*70}{Colors.ENDC}\n")
        
        # 关键配置
        key_sections = {
            'Model': ['arch', 'num_classes', 'use_gating', 'use_condition'],
            'Data': ['batch_size', 'num_workers', 'img_scale', 'crop_size'],
            'Training': ['max_epochs', 'lr', 'optimizer', 'param_scheduler'],
            'Device': ['cudnn_benchmark', 'launcher'],
        }
        
        for section, keys in key_sections.items():
            print(f"{Colors.YELLOW}[{section}]{Colors.ENDC}")
            
            for key in keys:
                value = ConfigPrinter._get_nested_value(cfg, key)
                if value is not None:
                    # 格式化输出
                    if isinstance(value, dict):
                        print(f"  {Colors.CYAN}{key}{Colors.ENDC}: {value.get('type', '?')}")
                    else:
                        print(f"  {Colors.CYAN}{key}{Colors.ENDC}: {value}")
            print()
    
    @staticmethod
    def _get_nested_value(cfg, key):
        """递归获取嵌套配置值"""
        for k, v in cfg.items():
            if isinstance(v, dict):
                if key in v:
                    return v[key]
                result = ConfigPrinter._get_nested_value(v, key)
                if result is not None:
                    return result
        return None