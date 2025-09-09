#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 測試數據生成器
這個腳本會創建用於測試的模擬數據集
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw
import yaml
import json

def create_classification_dataset(base_path="./test_classification_data", num_classes=5, images_per_class=20):
    """
    創建分類測試數據集
    """
    print(f"🎨 創建分類測試數據集: {num_classes}類，每類{images_per_class}張圖片")

    # 創建目錄結構
    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "val")

    class_names = [f"class_{i}" for i in range(num_classes)]

    for split_path in [train_path, val_path]:
        for class_name in class_names:
            class_dir = os.path.join(split_path, class_name)
            os.makedirs(class_dir, exist_ok=True)

    # 生成彩色圖片
    colors = [
        (255, 100, 100),  # 紅色
        (100, 255, 100),  # 綠色
        (100, 100, 255),  # 藍色
        (255, 255, 100),  # 黃色
        (255, 100, 255),  # 紫色
    ]

    for class_idx, class_name in enumerate(class_names):
        color = colors[class_idx % len(colors)]

        # 訓練集
        train_class_dir = os.path.join(train_path, class_name)
        for img_idx in range(images_per_class):
            img = Image.new('RGB', (224, 224), color)
            draw = ImageDraw.Draw(img)

            # 添加一些隨機形狀
            for _ in range(5):
                x1, y1 = random.randint(0, 200), random.randint(0, 200)
                x2, y2 = x1 + random.randint(10, 50), y1 + random.randint(10, 50)
                shape_color = tuple(random.randint(0, 255) for _ in range(3))
                draw.ellipse([x1, y1, x2, y2], fill=shape_color)

            img_path = os.path.join(train_class_dir, f"{class_name}_{img_idx:03d}.jpg")
            img.save(img_path)

        # 驗證集（較少圖片）
        val_class_dir = os.path.join(val_path, class_name)
        for img_idx in range(images_per_class // 4):
            img = Image.new('RGB', (224, 224), color)
            draw = ImageDraw.Draw(img)

            # 添加不同的形狀用於驗證
            for _ in range(3):
                x1, y1 = random.randint(0, 200), random.randint(0, 200)
                x2, y2 = x1 + random.randint(20, 60), y1 + random.randint(20, 60)
                shape_color = tuple(random.randint(50, 200) for _ in range(3))
                draw.rectangle([x1, y1, x2, y2], fill=shape_color)

            img_path = os.path.join(val_class_dir, f"{class_name}_val_{img_idx:03d}.jpg")
            img.save(img_path)

    print(f"✅ 分類數據集創建完成: {base_path}")
    return base_path

def create_yolo_dataset(base_path="./test_yolo_data", num_classes=3, num_images=50):
    """
    創建YOLO格式的物件偵測數據集
    """
    print(f"🎯 創建YOLO測試數據集: {num_classes}類，{num_images}張圖片")

    # 創建目錄結構
    images_train = os.path.join(base_path, "images", "train")
    images_val = os.path.join(base_path, "images", "val")
    labels_train = os.path.join(base_path, "labels", "train")
    labels_val = os.path.join(base_path, "labels", "val")

    for dir_path in [images_train, images_val, labels_train, labels_val]:
        os.makedirs(dir_path, exist_ok=True)

    # 類別名稱
    class_names = [f"object_{i}" for i in range(num_classes)]

    # 生成訓練數據
    for img_idx in range(num_images):
        # 創建圖片
        img = Image.new('RGB', (640, 640), (200, 200, 200))
        draw = ImageDraw.Draw(img)

        # 生成標註
        annotations = []
        num_objects = random.randint(1, 5)

        for obj_idx in range(num_objects):
            # 隨機類別
            class_id = random.randint(0, num_classes - 1)

            # 隨機位置和大小
            x_center = random.uniform(0.1, 0.9)
            y_center = random.uniform(0.1, 0.9)
            width = random.uniform(0.05, 0.3)
            height = random.uniform(0.05, 0.3)

            # 確保邊界框在圖片內
            x_center = max(width/2, min(1-width/2, x_center))
            y_center = max(height/2, min(1-height/2, y_center))

            # 繪製物件
            x1 = int((x_center - width/2) * 640)
            y1 = int((y_center - height/2) * 640)
            x2 = int((x_center + width/2) * 640)
            y2 = int((y_center + height/2) * 640)

            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            color = colors[class_id % len(colors)]
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

            # YOLO格式標註 (class_id x_center y_center width height)
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 保存圖片和標註
        split = "train" if img_idx < num_images * 0.8 else "val"

        img_path = os.path.join(base_path, "images", split, f"image_{img_idx:04d}.jpg")
        label_path = os.path.join(base_path, "labels", split, f"image_{img_idx:04d}.txt")

        img.save(img_path)

        with open(label_path, 'w') as f:
            f.write("\n".join(annotations))

    # 創建data.yaml
    data_yaml = {
        'train': './images/train',
        'val': './images/val',
        'nc': num_classes,
        'names': class_names
    }

    yaml_path = os.path.join(base_path, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✅ YOLO數據集創建完成: {base_path}")
    return base_path

def create_segmentation_dataset(base_path="./test_segmentation_data", num_classes=4, num_images=30):
    """
    創建YOLO格式的語義分割測試數據集
    """
    print(f"🖼️ 創建分割測試數據集: {num_classes}類，{num_images}張圖片")

    # 創建YOLO格式目錄結構
    images_train = os.path.join(base_path, "images", "train")
    images_val = os.path.join(base_path, "images", "val")
    labels_train = os.path.join(base_path, "labels", "train")
    labels_val = os.path.join(base_path, "labels", "val")

    for dir_path in [images_train, images_val, labels_train, labels_val]:
        os.makedirs(dir_path, exist_ok=True)

    # 類別名稱
    class_names = ['background'] + [f'class_{i}' for i in range(1, num_classes)]

    for img_idx in range(num_images):
        # 創建原始圖片
        img = Image.new('RGB', (640, 640), (128, 128, 128))
        draw = ImageDraw.Draw(img)

        # 生成分割標註
        annotations = []
        num_objects = random.randint(1, 3)  # 減少物件數量以便生成更簡單的分割

        for obj_idx in range(num_objects):
            # 隨機類別 (0是背景，所以從1開始)
            class_id = random.randint(1, num_classes - 1)

            # 隨機位置和大小
            x_center = random.uniform(0.2, 0.8)
            y_center = random.uniform(0.2, 0.8)
            width = random.uniform(0.1, 0.4)
            height = random.uniform(0.1, 0.4)

            # 確保邊界框在圖片內
            x_center = max(width/2, min(1-width/2, x_center))
            y_center = max(height/2, min(1-height/2, y_center))

            # 計算邊界框
            x1 = (x_center - width/2) * 640
            y1 = (y_center - height/2) * 640
            x2 = (x_center + width/2) * 640
            y2 = (y_center + height/2) * 640

            # 繪製物件
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            color = colors[class_id % len(colors)]
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

            # 生成多邊形分割標註 (簡化為矩形)
            # YOLO分割格式: class_id x1 y1 x2 y2 x3 y3 ... (多邊形頂點)
            # 這裡我們用矩形作為簡單的多邊形
            polygon_points = [
                x1/640, y1/640,  # 左上角
                x2/640, y1/640,  # 右上角
                x2/640, y2/640,  # 右下角
                x1/640, y2/640   # 左下角
            ]
            
            # 格式化為YOLO分割格式
            polygon_str = " ".join([f"{point:.6f}" for point in polygon_points])
            annotations.append(f"{class_id} {polygon_str}")

        # 保存圖片和標註
        split = "train" if img_idx < num_images * 0.8 else "val"

        img_path = os.path.join(base_path, "images", split, f"image_{img_idx:04d}.jpg")
        label_path = os.path.join(base_path, "labels", split, f"image_{img_idx:04d}.txt")

        img.save(img_path)

        with open(label_path, 'w') as f:
            f.write("\n".join(annotations))

    # 創建data.yaml
    data_yaml = {
        'train': './images/train',
        'val': './images/val',
        'nc': num_classes,
        'names': class_names
    }

    yaml_path = os.path.join(base_path, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✅ 分割數據集創建完成: {base_path}")
    return base_path

def main():
    """
    主函數 - 創建所有測試數據集
    """
    print("🎨 iVIT 2.0 測試數據生成器")
    print("=" * 50)

    # 創建輸出目錄
    base_dir = "./test_data"
    os.makedirs(base_dir, exist_ok=True)

    # 創建各種類型的測試數據集
    classification_path = create_classification_dataset(
        os.path.join(base_dir, "classification")
    )

    yolo_path = create_yolo_dataset(
        os.path.join(base_dir, "yolo")
    )

    segmentation_path = create_segmentation_dataset(
        os.path.join(base_dir, "segmentation")
    )

    print("\n🎉 所有測試數據集創建完成！")
    print(f"分類數據集: {classification_path}")
    print(f"YOLO數據集: {yolo_path}")
    print(f"分割數據集: {segmentation_path}")
    print("\n現在可以使用這些數據集來測試iVIT 2.0 SDK的各種功能")

if __name__ == "__main__":
    main()
