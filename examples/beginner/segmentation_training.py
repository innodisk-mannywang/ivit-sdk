#!/usr/bin/env python3
"""
iVIT 2.0 SDK - Segmentation 訓練範例
支援單卡和多卡訓練
"""

import os
import sys
import argparse

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ivit.trainer.segmentation import SegmentationTrainer


def run_segmentation_training(data_path: str, device: str, epochs: int = 50,
                            batch_size: int = 16, learning_rate: float = 0.01,
                            model_name: str = 'yolov8n-seg.pt', img_size: int = 640,
                            num_classes: int = None):
    """
    執行語義分割訓練
    
    Args:
        data_path: 資料集路徑 (YOLO 格式)
        device: 設備配置 ('0' 單卡, '0,1' 多卡)
        epochs: 訓練輪數
        batch_size: 批次大小
        learning_rate: 學習率
        model_name: 模型名稱
        img_size: 圖片尺寸
    """
    
    print("🚀 iVIT 2.0 Segmentation 訓練")
    print("=" * 50)
    print(f"📁 資料路徑: {data_path}")
    print(f"🔧 設備: {device}")
    print(f"📊 模型: {model_name}")
    print(f"📐 圖片尺寸: {img_size}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 批次大小: {batch_size}")
    print(f"📈 學習率: {learning_rate}")
    print("=" * 50)
    
    # 創建訓練器
    trainer = SegmentationTrainer(
        model_name=model_name,
        img_size=img_size,
        learning_rate=learning_rate,
        device=device
    )
    
    # 驗證資料集格式
    print("\n📋 驗證資料集格式...")
    if not trainer.validate_dataset_format(data_path):
        print("❌ 資料集格式不正確，請確保是 YOLO 格式")
        print("YOLO 格式結構:")
        print("dataset/")
        print("├── images/")
        print("│   ├── train/")
        print("│   └── val/")
        print("├── labels/")
        print("│   ├── train/")
        print("│   └── val/")
        print("└── data.yaml")
        return False
    
    print("✅ 資料集格式正確")
    
    # 開始訓練
    print("\n🎯 開始訓練...")
    try:
        results = trainer.train(
            dataset_path=data_path,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("\n🎉 訓練完成！")
        print("📊 最終結果:")
        if 'final_metrics' in results:
            for metric, value in results['final_metrics'].items():
                print(f"   {metric}: {value:.4f}")
        
        # 顯示模型保存位置
        if 'model_path' in results and results['model_path']:
            print(f"💾 模型已保存至: {results['model_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 訓練失敗: {str(e)}")
        return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 Segmentation 訓練')
    parser.add_argument('--data_path', type=str, required=True,
                       help='資料集路徑 (YOLO 格式)')
    parser.add_argument('--device', type=str, default='0',
                       help='設備配置 (單卡: "0", 多卡: "0,1,2,3")')
    parser.add_argument('--epochs', type=int, default=50,
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='學習率')
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt',
                       help='模型名稱 (yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt)')
    parser.add_argument('--img_size', type=int, default=640,
                       help='圖片尺寸')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='類別數量 (自動檢測如果未指定)')
    
    args = parser.parse_args()
    
    # 執行訓練
    success = run_segmentation_training(
        data_path=args.data_path,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model,
        img_size=args.img_size,
        num_classes=args.num_classes
    )
    
    if success:
        print("\n✅ 訓練成功完成！")
    else:
        print("\n❌ 訓練失敗！")
        sys.exit(1)


if __name__ == "__main__":
    main()
