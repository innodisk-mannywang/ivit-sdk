#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 初學者範例：物件偵測
這個範例展示如何使用YOLOv8進行物件偵測訓練
"""

from ivit.trainer.detection import DetectionTrainer
import os

def main():
    """
    初學者物件偵測範例 - 使用YOLOv8
    """

    # 步驟 1: 設定資料路徑
    # data_path = "path/to/your/yolo/dataset"
    data_path = "/home/ipa-genai/Workspace/Project/datasets/rock_paper_scissor"

    # YOLO資料集結構應該如下：
    # dataset/
    #   ├── images/
    #   │   ├── train/
    #   │   │   ├── img1.jpg
    #   │   │   └── img2.jpg
    #   │   └── val/
    #   │       ├── img1.jpg
    #   │       └── img2.jpg
    #   ├── labels/
    #   │   ├── train/
    #   │   │   ├── img1.txt
    #   │   │   └── img2.txt
    #   │   └── val/
    #   │       ├── img1.txt
    #   │       └── img2.txt
    #   └── data.yaml

    print("🚀 iVIT 2.0 初學者物件偵測範例")
    print("=" * 50)

    # 步驟 2: 創建訓練器（直接使用 DetectionTrainer 的參數）
    trainer = DetectionTrainer(
        model_name='yolov8n.pt',
        img_size=640,
        learning_rate=0.01,
        device='0,1'
    )

    # 步驟 4: 驗證資料集格式
    print("\n📋 驗證資料集格式...")
    is_valid = trainer.validate_dataset_format(data_path)
    if not is_valid:
        print("❌ 資料集格式不正確，請檢查YOLO格式")
        return

    print("✅ 資料集格式正確")

    # 步驟 5: 開始訓練
    print("\n🎯 開始訓練...")
    trainer.train(
        dataset_path=data_path,
        epochs=50,
        batch_size=16
    )

    print("\n🎉 訓練完成！")
    print("模型已儲存（請參考訓練輸出目錄）")

if __name__ == "__main__":
    # 確保資料路徑存在
    print("請確保您的YOLO資料集路徑正確，然後執行此程式")
    print("如需測試，請使用 python create_test_data.py 創建測試資料")
    main()  # 取消註解來執行訓練
