#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 初學者範例：圖像分類
這個範例展示如何使用最簡單的方式進行圖像分類訓練
"""

from ivit.trainer.classification import ClassificationTrainer
import os

def main():
    """
    初學者分類範例 - 使用預設設定
    """

    # 步驟 1: 設定資料路徑（需為 torchvision ImageFolder 結構）
    # 結構需為：
    # dataset/
    #   ├── train/
    #   │   ├── class1/
    #   │   │   ├── img1.jpg
    #   │   │   └── img2.jpg
    #   │   └── class2/
    #   │       ├── img1.jpg
    #   │       └── img2.jpg
    #   └── val/
    #       ├── class1/
    #       └── class2/
    data_path = "/home/ipa-genai/Workspace/Project/datasets"

    print("🚀 iVIT 2.0 初學者分類範例")
    print("=" * 50)

    # 讀取環境變數
    epochs = int(os.getenv("EPOCHS", "10"))
    run_name = os.getenv("RUN_NAME", "classification_output")

    # 步驟 2: 創建訓練器（使用預設 resnet18 與預訓練權重）
    trainer = ClassificationTrainer(
        model_name='resnet18',
        num_classes=None,          # 若未指定，將自動由資料集偵測
        pretrained=True,
        learning_rate=0.001,
        device='auto'
    )

    # 步驟 3: 開始訓練
    print("\n🎯 開始訓練...")
    output_dir = os.path.join("./", run_name)
    os.makedirs(output_dir, exist_ok=True)
    trainer.train(
        dataset_path=data_path,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        save_path=os.path.join(output_dir, "best.pt")
    )

    print("\n🎉 訓練完成！")
    print(f"模型已儲存至: {os.path.join(output_dir, 'best.pt')}")

if __name__ == "__main__":
    # 確保資料路徑存在
    print("請確保您的資料集路徑正確，然後執行此程式")
    print("如需測試，請使用 python create_test_data.py 創建測試資料")
    main()  # 取消註解來執行訓練
