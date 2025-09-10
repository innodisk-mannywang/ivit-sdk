#!/usr/bin/env python3
"""
iVIT 2.0 SDK - Classification 訓練範例
支援單卡和多卡訓練
"""

import os
import sys
import argparse

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ivit.trainer.classification import ClassificationTrainer


def run_classification_training(data_path: str, device: str, epochs: int = 50, 
                              batch_size: int = 16, learning_rate: float = 0.01,
                              model_name: str = 'resnet18', img_size: int = 224,
                              num_classes: int = 3):
    """
    執行分類訓練
    
    Args:
        data_path: 資料集路徑 (ImageFolder 格式)
        device: 設備配置 ('0' 單卡, '0,1' 多卡)
        epochs: 訓練輪數
        batch_size: 批次大小
        learning_rate: 學習率
        model_name: 模型名稱
        img_size: 圖片尺寸
        num_classes: 類別數量
    """
    
    print("🚀 iVIT 2.0 Classification 訓練")
    print("=" * 50)
    print(f"📁 資料路徑: {data_path}")
    print(f"🔧 設備: {device}")
    print(f"📊 模型: {model_name}")
    print(f"📐 圖片尺寸: {img_size}")
    print(f"🏷️ 類別數量: {num_classes}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 批次大小: {batch_size}")
    print(f"📈 學習率: {learning_rate}")
    print("=" * 50)
    
    # 創建訓練器
    trainer = ClassificationTrainer(
        model_name=model_name,
        img_size=img_size,
        num_classes=num_classes,
        learning_rate=learning_rate,
        device=device
    )
    
    # 驗證資料集
    print("\n📋 驗證資料集格式...")
    if not os.path.exists(data_path):
        print(f"❌ 資料集路徑不存在: {data_path}")
        return False
    
    print("✅ 資料集路徑存在")
    
    # 創建模型保存路徑
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"models/classification_{model_name}_{timestamp}.pth"
    
    # 確保模型目錄存在
    os.makedirs("models", exist_ok=True)
    
    # 開始訓練
    print("\n🎯 開始訓練...")
    try:
        results = trainer.train(
            dataset_path=data_path,
            epochs=epochs,
            batch_size=batch_size,
            save_path=model_save_path
        )
        
        print("\n🎉 訓練完成！")
        print("📊 最終結果:")
        if 'final_metrics' in results:
            for metric, value in results['final_metrics'].items():
                print(f"   {metric}: {value:.4f}")
        
        # 顯示模型保存位置
        if os.path.exists(model_save_path):
            print(f"\n💾 模型已保存至: {os.path.abspath(model_save_path)}")
            
            # 檢查是否產生了 class_names.json
            model_name = os.path.splitext(os.path.basename(model_save_path))[0]
            class_names_path = os.path.join("models", f"{model_name}_class_names.json")
            if os.path.exists(class_names_path):
                print(f"📋 類別名稱已保存至: {os.path.abspath(class_names_path)}")
                print("   您可以在推理時使用此檔案作為 --class_names 參數")
        
        return True
        
    except Exception as e:
        print(f"❌ 訓練失敗: {str(e)}")
        return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 Classification 訓練')
    parser.add_argument('--data_path', type=str, required=True,
                       help='資料集路徑 (ImageFolder 格式)')
    parser.add_argument('--device', type=str, default='0',
                       help='設備配置 (單卡: "0", 多卡: "0,1,2,3")')
    parser.add_argument('--epochs', type=int, default=50,
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='學習率')
    parser.add_argument('--model', type=str, default='resnet18',
                       help='模型名稱 (resnet18, resnet50, efficientnet_b0)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='圖片尺寸')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='類別數量')
    
    args = parser.parse_args()
    
    # 執行訓練
    success = run_classification_training(
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
