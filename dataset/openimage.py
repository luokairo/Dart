import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class DatasetJson(Dataset):
    def __init__(self, data_path, caption_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        
        # 加载图片路径列表
        json_path = os.path.join(data_path, 'image_paths.json')
        assert os.path.exists(json_path), f"please first run: python3 tools/openimage_json.py"
        with open(json_path, 'r') as f:
            self.image_paths = json.load(f)

        # 加载 caption 数据
        self.image_id_to_caption = self.load_captions(caption_path)
        
        # 只保留有对应 caption 的图像
        self.filtered_image_paths = [
            image_path for image_path in self.image_paths
            if os.path.splitext(os.path.basename(image_path))[0] in self.image_id_to_caption
        ]

    def load_captions(self, caption_path):
        """加载 caption 文件并生成 image_id -> caption 的映射"""
        image_id_to_caption = {}
        with open(caption_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                image_id = data["image_id"]
                caption = data["caption"]
                image_id_to_caption[image_id] = caption
        return image_id_to_caption

    def __len__(self):
        # 返回仅包含有 caption 的图像数量
        return len(self.filtered_image_paths)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def getdata(self, idx):
        # 获取当前图像路径
        image_path = self.filtered_image_paths[idx]
        # 获取图像 ID
        image_id = os.path.splitext(os.path.basename(image_path))[0]  # 提取不带扩展名的文件名
        image_path_full = os.path.join(self.data_path, image_path)

        # 打开图片
        image = Image.open(image_path_full).convert('RGB')
        
        # 获取对应的 caption
        label = self.image_id_to_caption.get(image_id, "a photo of")  # 默认值 "a photo of" 如果找不到匹配的 caption

        if self.transform:
            image = self.transform(image)
        
        return image, label


def build_openimage(args, transform):
    return DatasetJson(args.data_path, args.caption_path, transform)

