
"""
FaceRecognition-PyTorch
train.py (Windows / PyTorch Final Version)

Author: Chen Yongfang
"""

import os
import csv
import copy
import random
import time
from multiprocessing import freeze_support

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD = True
except Exception:
    USE_TENSORBOARD = False

from dataset import create_dataloader
from model import get_model

IMAGE_SIZE = 160
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SELF_DIR = os.path.join(DATASET_DIR, "chenyongfang")
UNKNOWN_DIR = os.path.join(DATASET_DIR, "unknown")

MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULT_DIR = os.path.join(BASE_DIR, "results")
RUN_DIR = os.path.join(BASE_DIR, "runs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
writer = SummaryWriter("runs/FaceRecognition") if USE_TENSORBOARD else None


def generate_unknown_dataset():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)

    if len([f for f in os.listdir(UNKNOWN_DIR) if f.endswith(".jpg")]) > 200:
        print("Unknown 数据集已经存在。")
        return

    print("生成 Unknown 数据集...")

    images = [f for f in os.listdir(SELF_DIR) if f.endswith(".jpg")]

    for idx, name in enumerate(images):
        img = cv2.imread(os.path.join(SELF_DIR, name))
        if img is None:
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        choice = random.randint(0, 5)

        if choice == 0:
            img = cv2.GaussianBlur(img, (9, 9), 0)

        elif choice == 1:
            cv2.rectangle(img, (20, 40), (140, 70), (0, 0, 0), -1)

        elif choice == 2:
            M = cv2.getRotationMatrix2D((80, 80), random.randint(-45, 45), 1)
            img = cv2.warpAffine(img, M, (160, 160))

        elif choice == 3:
            x = random.randint(0, 20)
            y = random.randint(0, 20)
            img = img[y:y+140, x:x+140]
            img = cv2.resize(img, (160, 160))

        elif choice == 4:
            alpha = random.uniform(0.5, 1.5)
            beta = random.randint(-30, 30)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        else:
            noise = np.random.normal(0, 20, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(UNKNOWN_DIR, f"{idx:04d}.jpg"), img)


class EarlyStopping:

    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def step(self, loss, model):
        save_path = os.path.join(MODEL_DIR, "best_model.pth")

        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0

            try:
                # 如果旧文件存在，先删除（Windows 有时会锁文件）
                if os.path.exists(save_path):
                    os.remove(save_path)

                torch.save(model.state_dict(), save_path)
                print(f"✅ 保存最佳模型：{save_path}")

            except Exception as e:
                print("❌ 保存模型失败")
                print("路径：", save_path)
                print("错误：", e)
                raise

        else:
            self.counter += 1
            print(f"EarlyStopping {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.stop = True


def accuracy(outputs, labels):
    pred = torch.argmax(outputs, dim=1)
    return (pred == labels).float().mean().item()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0
    acc_sum = 0

    bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(outputs, labels)

        loss_sum += loss.item()
        acc_sum += acc

        bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc*100:.2f}%")

    return loss_sum / len(loader), acc_sum / len(loader)


def valid_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = 0
    acc_sum = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            acc_sum += accuracy(outputs, labels)

    return loss_sum / len(loader), acc_sum / len(loader)


def test(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            pred = torch.argmax(model(images), dim=1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"测试准确率：{acc*100:.2f}%")
    return acc


def save_curve(train_values, valid_values, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(train_values, label="Train")
    plt.plot(valid_values, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename), dpi=300)
    plt.close()


def main():

    generate_unknown_dataset()

    train_loader, valid_loader, test_loader, label_dict = create_dataloader()

    num_classes = max(len(label_dict), 2)

    print("="*60)
    print("类别：", label_dict)
    print("Train:", len(train_loader))
    print("Valid:", len(valid_loader))
    print("Test :", len(test_loader))
    print("="*60)

    model, device = get_model(num_classes=num_classes, force_cpu=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    early_stop = EarlyStopping(PATIENCE)

    train_loss_list = []
    valid_loss_list = []

    train_acc_list = []
    valid_acc_list = []

    csv_path = os.path.join(RESULT_DIR, "training_log.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer_csv = csv.writer(f)

        writer_csv.writerow([
            "Epoch",
            "TrainLoss",
            "ValidLoss",
            "TrainAcc",
            "ValidAcc",
            "LR"
        ])

        print("开始训练 FaceCNN")

        start = time.time()

        for epoch in range(EPOCHS):

            print(f"\nEpoch {epoch+1}/{EPOCHS}")

            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            valid_loss, valid_acc = valid_epoch(
                model,
                valid_loader,
                criterion,
                device
            )

            scheduler.step(valid_loss)

            lr = optimizer.param_groups[0]["lr"]

            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)

            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

            writer_csv.writerow([
                epoch+1,
                train_loss,
                valid_loss,
                train_acc,
                valid_acc,
                lr
            ])

            if USE_TENSORBOARD:
                writer.add_scalar("Loss/Train", train_loss, epoch+1)
                writer.add_scalar("Loss/Validation", valid_loss, epoch+1)
                writer.add_scalar("Accuracy/Train", train_acc, epoch+1)
                writer.add_scalar("Accuracy/Validation", valid_acc, epoch+1)

            print(f"Train Loss : {train_loss:.4f}")
            print(f"Valid Loss : {valid_loss:.4f}")
            print(f"Train Acc  : {train_acc*100:.2f}%")
            print(f"Valid Acc  : {valid_acc*100:.2f}%")
            print(f"LR         : {lr:.6f}")

            early_stop.step(valid_loss, model)

            if early_stop.stop:
                print("Early Stopping Triggered.")
                break

        print(f"\n训练完成，总耗时 {time.time()-start:.2f}s")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last_model.pth"))

    save_curve(train_loss_list, valid_loss_list, "Loss", "loss.png")
    save_curve(train_acc_list, valid_acc_list, "Accuracy", "accuracy.png")

    test(model, test_loader, device)

    if USE_TENSORBOARD:
        writer.close()

    print("="*60)
    print("生成文件：")
    print("model/best_model.pth")
    print("model/last_model.pth")
    print("results/loss.png")
    print("results/accuracy.png")
    print("results/training_log.csv")
    print("="*60)


if __name__ == "__main__":
    freeze_support()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    main()
