import os
import json
import numpy as np
from PIL import Image
from torchvision import transforms

from dart.dataset.augmentation import random_crop_arr
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder



class DatasetJson(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        json_path = os.path.join(data_path, 'image_paths.json')
        assert os.path.exists(json_path), f"please first run: python3 tools/openimage_json.py"
        with open(json_path, 'r') as f:
            self.image_paths = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')
    
    def getdata(self, idx):
        image_path = self.image_paths[idx]
        image_path_full = os.path.join(self.data_path, image_path)
        image = Image.open(image_path_full).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)

data_path = '/fs/scratch/PAS2473/ICML2025/dataset/openimage'
image_size = 1024

transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

dataset = DatasetJson(data_path, transform)
print(len(dataset))

# 将 Dataset 转换为 DataLoader
batch_size = 1  # 可以根据需要调整批次大小
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 随机打印前面3个样本
for i, (image, label) in enumerate(dataloader):
    if i >= 3:
        break
    print(f"Sample {i+1}:")
    print(f"Image tensor shape: {image.shape}")
    print(f"Label tensor: {label}")
