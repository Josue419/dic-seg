import torch



ckpt = torch.load('/mnt/e/projects/dic/best_val_mIoU_epoch_2.pth')
print(ckpt.keys())  # 应只有 ['state_dict', 'optimizer', 'meta', ...]
for k, v in ckpt.items():
    if isinstance(v, dict):
        for kk, vv in v.items():
            if torch.is_tensor(vv):
                print(f"{k}.{kk}: {vv.shape}, {vv.numel() * 4 / 1e9:.2f} GB")