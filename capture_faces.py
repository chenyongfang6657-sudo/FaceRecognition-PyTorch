import cv2
import os
import numpy as np
from datetime import datetime

# ==========================
# 配置参数
# ==========================
IMAGE_SIZE = 160
SAVE_NUMBER = 230
PERSON_NAME = "chenyongfang"  # 修改成采集对象名字s
DATASET_DIR = os.path.join("dataset", PERSON_NAME)

os.makedirs(DATASET_DIR, exist_ok=True)

# OpenCV 自带人脸检测模型
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("无法打开摄像头！")

count = len(os.listdir(DATASET_DIR))

print("=" * 50)
print("PyTorch 人脸识别数据采集")
print(f"采集对象：{PERSON_NAME}")
print(f"保存目录：{DATASET_DIR}")
print("快捷键：")
print("S —— 保存图片")
print("ESC —— 退出")
print("=" * 50)


def augment_face(face):
    """返回增强后的图片列表"""
    aug_images = []

    # 原图
    aug_images.append(face)

    # 镜像
    aug_images.append(cv2.flip(face, 1))

    # 旋转 +10°
    M = cv2.getRotationMatrix2D((80, 80), 10, 1.0)
    aug_images.append(cv2.warpAffine(face, M, (160, 160)))

    # 旋转 -10°
    M = cv2.getRotationMatrix2D((80, 80), -10, 1.0)
    aug_images.append(cv2.warpAffine(face, M, (160, 160)))

    # 增加亮度
    bright = cv2.convertScaleAbs(face, alpha=1.2, beta=30)
    aug_images.append(bright)

    # 降低亮度
    dark = cv2.convertScaleAbs(face, alpha=0.8, beta=-20)
    aug_images.append(dark)

    return aug_images


while True:
    success, frame = camera.read()
    if not success:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    show = frame.copy()

    cv2.putText(show,
                f"Collected: {count}/{SAVE_NUMBER}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    for (x, y, w, h) in faces:
        cv2.rectangle(show, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))

        preview = face.copy()
        cv2.putText(preview,
                    "Press S to Save",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)

        cv2.imshow("Face Preview", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            images = augment_face(face)

            for img in images:
                filename = f"{count:04d}.jpg"
                cv2.imwrite(os.path.join(DATASET_DIR, filename), img)
                count += 1

            print(f"已保存 {len(images)} 张增强图片，总数：{count}")

        if count >= SAVE_NUMBER:
            print("采集完成！")
            camera.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow("Face Collection", show)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()