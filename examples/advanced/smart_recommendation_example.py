#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 進階範例：智能參數推薦
這個範例展示如何使用智能推薦系統自動選擇最佳訓練參數
"""

import sys
import os
import argparse
# 添加專案根目錄到 Python 路徑
if '__file__' in globals():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
else:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('smart_recommendation.py'))))
sys.path.insert(0, project_root)

from ivit.trainer.classification import ClassificationConfig, ClassificationTrainer
from ivit.trainer.detection import DetectionConfig, DetectionTrainer
from ivit.utils.dataset_analyzer import DatasetAnalyzer
from ivit.utils.smart_recommendation import SmartRecommendationEngine

def classification_with_smart_recommendation(data_path=None):
    """
    使用智能推薦進行分類訓練
    
    Args:
        data_path: 資料集路徑，如果為 None 則讓使用者選擇
    """
    print("🧠 智能推薦 - 分類任務")
    print("=" * 50)

    # 步驟 1: 獲取資料集路徑
    if data_path is None:
        print("📁 選擇資料集來源：")
        print("   1. 使用預設測試資料集")
        print("   2. 輸入自定義資料集路徑")
        
        choice = input("請選擇 (1 或 2): ").strip()
        
        if choice == "1":
            data_path = "../../test_data/classification"
            print(f"✅ 使用預設測試資料集: {data_path}")
        elif choice == "2":
            data_path = input("請輸入分類資料集路徑 (包含 train/ 和 val/ 目錄): ").strip()
            if not os.path.exists(data_path):
                print(f"❌ 路徑不存在: {data_path}")
                return
            print(f"✅ 使用自定義資料集: {data_path}")
        else:
            print("❌ 無效的選擇，使用預設測試資料集")
            data_path = "../../test_data/classification"
    else:
        if not os.path.exists(data_path):
            print(f"❌ 路徑不存在: {data_path}")
            return
        print(f"✅ 使用指定資料集: {data_path}")
    
    # 步驟 2: 分析資料集
    analyzer = DatasetAnalyzer()

    print("📊 分析資料集特徵...")
    stats = analyzer.analyze_classification_dataset(f"{data_path}/train")

    print(f"   類別數量: {stats['num_classes']}")
    print(f"   總圖片數: {stats['total_samples']}")
    print(f"   平均圖片大小: {stats['image_stats']['mean_width']:.0f}x{stats['image_stats']['mean_height']:.0f}")
    print(f"   資料集複雜度: {stats['complexity_score']:.2f}")

    # 步驟 2: 獲取智能推薦
    recommender = SmartRecommendationEngine()
    recommendations = recommender.get_recommendations('classification', stats)

    print("\n🎯 智能推薦結果:")
    print(f"   推薦模型: {recommendations['model']}")
    print(f"   學習率: {recommendations['learning_rate']}")
    print(f"   批次大小: {recommendations['batch_size']}")
    print(f"   訓練輪數: {recommendations['epochs']}")
    print(f"   推薦原因: {recommendations['reasoning']}")

    # 步驟 3: 使用推薦參數創建配置
    config = ClassificationConfig(
        train_data=f"{data_path}/train",
        val_data=f"{data_path}/val",
        num_classes=stats['num_classes'],
        model_name=recommendations['model'],
        learning_rate=recommendations['learning_rate'],
        batch_size=recommendations['batch_size'],
        epochs=recommendations['epochs'],
        output_dir="./smart_classification_output"
    )

    # 步驟 4: 執行訓練
    print("\n🚀 使用智能推薦參數開始訓練...")
    trainer = ClassificationTrainer(config)
    # 注意：實際訓練需要提供資料集路徑
    # trainer.train(dataset_path=f"{data_path}/train")
    print("💡 提示：要開始實際訓練，請取消註解上面的 train() 調用")

    print("\n✅ 智能推薦訓練完成！")

def detection_with_smart_recommendation(data_yaml=None):
    """
    使用智能推薦進行物件偵測訓練
    
    Args:
        data_yaml: YOLO資料集的 data.yaml 檔案路徑，如果為 None 則讓使用者選擇
    """
    print("\n🧠 智能推薦 - 物件偵測任務")
    print("=" * 50)

    # 步驟 1: 獲取資料集路徑
    if data_yaml is None:
        print("📁 選擇資料集來源：")
        print("   1. 使用預設測試資料集")
        print("   2. 輸入自定義資料集路徑")
        
        choice = input("請選擇 (1 或 2): ").strip()
        
        if choice == "1":
            data_yaml = "../../test_data/detection/data.yaml"
            print(f"✅ 使用預設測試資料集: {data_yaml}")
        elif choice == "2":
            data_yaml = input("請輸入YOLO資料集的 data.yaml 檔案路徑: ").strip()
            if not os.path.exists(data_yaml):
                print(f"❌ 檔案不存在: {data_yaml}")
                return
            print(f"✅ 使用自定義資料集: {data_yaml}")
        else:
            print("❌ 無效的選擇，使用預設測試資料集")
            data_yaml = "../../test_data/detection/data.yaml"
    else:
        if not os.path.exists(data_yaml):
            print(f"❌ 檔案不存在: {data_yaml}")
            return
        print(f"✅ 使用指定資料集: {data_yaml}")
    
    # 步驟 2: 分析YOLO資料集
    analyzer = DatasetAnalyzer()

    print("📊 分析YOLO資料集...")
    stats = analyzer.analyze_yolo_dataset(data_yaml)

    print(f"   類別數量: {stats['num_classes']}")
    print(f"   總圖片數: {stats['total_images']}")
    print(f"   平均物件數/圖: {stats['bbox_stats']['mean_boxes_per_image']:.2f}")
    print(f"   資料集複雜度: {stats['complexity_score']:.2f}")

    # 步驟 2: 獲取智能推薦
    recommender = SmartRecommendationEngine()
    recommendations = recommender.get_recommendations('detection', stats)

    print("\n🎯 智能推薦結果:")
    print(f"   推薦模型: {recommendations['model']}")
    print(f"   學習率: {recommendations['learning_rate']}")
    print(f"   批次大小: {recommendations['batch_size']}")
    print(f"   訓練輪數: {recommendations['epochs']}")
    print(f"   推薦原因: {recommendations['reasoning']}")

    # 步驟 3: 使用推薦參數創建配置
    config = DetectionConfig(
        data_yaml=data_yaml,
        model_name=recommendations['model'],
        learning_rate=recommendations['learning_rate'],
        batch_size=recommendations['batch_size'],
        epochs=recommendations['epochs'],
        output_dir="./smart_detection_output"
    )

    # 步驟 4: 執行訓練
    print("\n🚀 使用智能推薦參數開始訓練...")
    trainer = DetectionTrainer(config)
    # 注意：實際訓練需要提供資料集路徑
    # trainer.train(dataset_path=data_yaml)
    print("💡 提示：要開始實際訓練，請取消註解上面的 train() 調用")

    print("\n✅ 智能推薦訓練完成！")

def main():
    """
    主函數 - 展示智能推薦功能
    """
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='iVIT 2.0 智能推薦系統範例')
    parser.add_argument('--task', type=str, choices=['classification', 'detection'], 
                       help='任務類型: classification 或 detection')
    parser.add_argument('--data_path', type=str, 
                       help='資料集路徑 (分類: 包含 train/ 和 val/ 的目錄, 偵測: data.yaml 檔案路徑)')
    parser.add_argument('--interactive', action='store_true', 
                       help='互動模式，讓使用者選擇任務類型和資料集')
    
    args = parser.parse_args()
    
    print("🚀 iVIT 2.0 智能推薦系統範例")
    print("=" * 60)
    
    # 如果指定了 --interactive 或沒有指定任何參數，則使用互動模式
    if args.interactive or (not args.task and not args.data_path):
        # 選擇任務類型
        task_type = input("選擇任務類型 (1: 分類, 2: 物件偵測): ")
        
        if task_type == "1":
            classification_with_smart_recommendation()
        elif task_type == "2":
            detection_with_smart_recommendation()
        else:
            print("❌ 無效的選擇")
    else:
        # 使用命令行參數
        if args.task == "classification":
            classification_with_smart_recommendation(args.data_path)
        elif args.task == "detection":
            detection_with_smart_recommendation(args.data_path)
        else:
            print("❌ 請指定有效的任務類型 (--task classification 或 detection)")

if __name__ == "__main__":
    print("🧠 此範例展示如何使用智能推薦系統")
    print("系統會自動分析您的資料集並推薦最佳參數")
    print("請確保資料集路徑正確後執行")
    print("\n📖 使用方式：")
    print("  互動模式: python3 smart_recommendation_example.py")
    print("  分類任務: python3 smart_recommendation_example.py --task classification --data_path /path/to/dataset")
    print("  偵測任務: python3 smart_recommendation_example.py --task detection --data_path /path/to/data.yaml")
    print("  查看幫助: python3 smart_recommendation_example.py --help")
    print()
    main()
