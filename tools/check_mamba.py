# check_mamba_cuda_compat.py
import torch
try:
    from mamba_ssm import selective_scan_fn
    print("✅ mamba_ssm loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load mamba_ssm: {e}")
    exit(1)

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA version (used to build): {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current GPU CUDA capability: {torch.cuda.get_device_capability()}")
    print(f"Current CUDA driver version: {torch.version.cuda} (PyTorch built with)")

# 尝试运行一个最小 Mamba 前向
with torch.no_grad():
    x = torch.randn(1, 64, 768).cuda()
    from mamba_ssm.modules.mamba_simple import Mamba
    model = Mamba(d_model=768, d_state=16).cuda()
    y = model(x)
    print(f"✅ Mamba forward test passed. Output shape: {y.shape}")