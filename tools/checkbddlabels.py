import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# BDD100K / Cityscapes 标准映射
CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain',
           'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
           'motorcycle', 'bicycle')

def check_classes(label_dir):
    print(f"正在检查标签目录: {label_dir}")
    files = glob(os.path.join(label_dir, '*.png'))
    
    # 统计每个类别出现的像素总数
    pixel_counts = {i: 0 for i in range(256)} 
    # 统计包含该类别的图片数量
    img_counts = {i: 0 for i in range(256)}
    
    # 为了速度，只随机抽查 1000 张，或者去掉 [:1000] 跑全量
    for f in tqdm(files): 
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None: continue
        
        # 获取该图中出现的所有唯一ID
        unique_ids, counts = np.unique(img, return_counts=True)
        
        for uid, cnt in zip(unique_ids, counts):
            pixel_counts[uid] += cnt
            img_counts[uid] += 1

    print("\n" + "="*40)
    print(f"{'Class ID':<10} {'Class Name':<15} {'Images':<10} {'Pixel Ratio'}")
    print("="*40)
    
    total_pixels = sum(pixel_counts.values())
    
    for i in range(19): # 0-18 是有效类别
        name = CLASSES[i]
        count = img_counts[i]
        ratio = (pixel_counts[i] / total_pixels) * 100
        print(f"{i:<10} {name:<15} {count:<10} {ratio:.4f}%")
        
        if count == 0:
            print(f"⚠️ 警告: 类别 {name} (ID={i}) 在抽样数据中从未出现！")

    print("-"*40)
    print(f"255 (Ignore) 出现次数: {img_counts[255]}")
    print("="*40)

if __name__ == '__main__':
    # 修改为你的 mask 路径
    label_path = '/root/projects/data/bdd100k_fixed/labels/train' 
    check_classes(label_path)