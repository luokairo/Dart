import os
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# 假设你的数据路径和caption路径如下
data_path = "/fs/scratch/PAS2473/ICML2025/dataset/openimage"  # 替换成实际的图像数据路径
caption_path = "/fs/scratch/PAS2473/ICML2025/dataset/openimage/label/merged_open_images_captions.jsonl"  # 替换成实际的caption文件路径

# 定义图像预处理操作（如果有需要的话）
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 可以根据需要调整大小
    transforms.ToTensor(),
])

# 加载Dataset
from openimage import DatasetJson  # 确保这行代码指向你的 DatasetJson 类

dataset = DatasetJson(data_path, caption_path, transform)

# 打印 dataset 的长度
print(f"Dataset length: {len(dataset)}")

# 打印前5行数据
for i in range(5):
    img, label = dataset[i]
    print(f"Image {i + 1}:")
    print(f"Label: {label}")
    print(f"Image size: {img.size() if isinstance(img, torch.Tensor) else 'Not a tensor'}\n")

# 如果需要，也可以使用 DataLoader 来批量加载数据
batch_size = 3
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 打印一个批次的数据
for batch_idx, (imgs, labels) in enumerate(loader):
    if batch_idx == 0:
        print("Batch data (first batch):")
        print(f"Labels: {labels}")
        print(labels.shape)
        print(f"Image batch size: {imgs.size()}")
        break  # 只打印一个批次数据