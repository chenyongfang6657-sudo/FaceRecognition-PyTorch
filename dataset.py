"""
dataset.py
PyTorch 人脸识别数据集加载模块
作者：Chen Yongfang
"""

import os
import random
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ==========================================
# 配置
# ==========================================

IMAGE_SIZE = 160
BATCH_SIZE = 32
DATASET_PATH = "dataset"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ==========================================
# 数据增强（训练集）
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.GaussianBlur(
        kernel_size=(3, 3),
        sigma=(0.1, 1.0)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 验证/测试集增强
# ==========================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================
# 扫描数据集
# ==========================================

def scan_dataset(root_dir="dataset"):
    image_paths = []
    labels = []

    persons = []

    # 只保留有图片的文件夹
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if len(image_files) > 0:
            persons.append(folder)

    # 保证 chenyongfang 永远是标签0
    if "chenyongfang" in persons:
        persons.remove("chenyongfang")
        persons.insert(0, "chenyongfang")

    label_dict = {person: idx for idx, person in enumerate(persons)}

    for person in persons:
        folder_path = os.path.join(root_dir, person)

        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(folder_path, file))
                labels.append(label_dict[person])

    return image_paths, labels, label_dict

# ==========================================
# Dataset 类
# ==========================================

class FaceDataset(Dataset):

    def __init__(
            self,
            image_paths: List[str],
            labels: List[int],
            transform=None
    ):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):

        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        return image, label

# ==========================================
# 创建 DataLoader
# ==========================================

def create_dataloader():

    image_paths, labels, label_dict = scan_dataset()

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_labels
    )

    train_dataset = FaceDataset(
        train_paths,
        train_labels,
        transform=train_transform
    )

    valid_dataset = FaceDataset(
        valid_paths,
        valid_labels,
        transform=test_transform
    )

    test_dataset = FaceDataset(
        test_paths,
        test_labels,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
        label_dict
    )

# ==========================================
# 查看数据集信息
# ==========================================

def dataset_info():

    image_paths, labels, label_dict = scan_dataset()

    print("="*60)
    print("PyTorch 人脸识别数据集信息")
    print("="*60)

    print(f"总图片数量：{len(image_paths)}")
    print(f"类别数量：{len(label_dict)}")

    reverse = {v:k for k,v in label_dict.items()}

    for idx in reverse:

        count = labels.count(idx)

        print(f"[{idx}] {reverse[idx]} -> {count} 张")

    print("="*60)

# ==========================================
# 测试 Dataset
# ==========================================

if __name__ == "__main__":

    dataset_info()

    train_loader, valid_loader, test_loader, label_dict = create_dataloader()

    print(f"训练 Batch 数：{len(train_loader)}")
    print(f"验证 Batch 数：{len(valid_loader)}")
    print(f"测试 Batch 数：{len(test_loader)}")

    images, labels = next(iter(train_loader))

    print("图片 Tensor Shape：", images.shape)
    print("标签 Tensor Shape：", labels.shape)
    print("标签示例：", labels[:10])