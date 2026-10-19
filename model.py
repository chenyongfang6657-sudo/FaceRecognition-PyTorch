"""
model.py
PyTorch 人脸识别 CNN 模型（最终版）
适配 dataset.py / train.py
作者：Chen Yongfang
"""

import torch
import torch.nn as nn

IMAGE_SIZE = 160


# =====================================================
# Conv Block
# =====================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


# =====================================================
# CNN
# =====================================================
class FaceCNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # 至少两类（本人 / Unknown）
        self.num_classes = max(num_classes, 2)

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

        # 初始化权重
        self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    # -------------------------------
    # He(Kaiming) 初始化
    # -------------------------------
    def initialize_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)


# =====================================================
# 获取模型
# =====================================================
def get_model(num_classes, force_cpu=True):
    """
    force_cpu=True：RTX5060 当前 PyTorch 不兼容，先强制 CPU。
    后续升级 PyTorch 后改成 False 即可。
    """

    if force_cpu:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)

            if capability[0] >= 12:
                print("⚠ RTX50 系列当前 PyTorch CUDA 不兼容，自动切换 CPU。")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model = FaceCNN(num_classes=num_classes)
    model = model.to(device)

    return model, device


# =====================================================
# 打印模型信息
# =====================================================
def model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 60)
    print("FaceCNN Model Summary")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print("=" * 60)


# =====================================================
# 测试
# =====================================================
if __name__ == "__main__":

    model, device = get_model(num_classes=2, force_cpu=True)

    model_summary(model)

    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    output = model(dummy)

    print("输入尺寸 :", dummy.shape)
    print("输出尺寸 :", output.shape)
    print("Device :", device)