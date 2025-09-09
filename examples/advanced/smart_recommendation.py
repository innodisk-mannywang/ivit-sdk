#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 進階範例：智能參數推薦
這個範例展示如何使用智能推薦系統自動選擇最佳訓練參數
"""

from ivit.trainer.classification import ClassificationConfig, ClassificationTrainer
from ivit.trainer.detection import DetectionConfig, DetectionTrainer
from ivit.utils.dataset_analyzer import DatasetAnalyzer
from ivit.utils.smart_recommendation import SmartRecommendationEngine
import os

def classification_with_smart_recommendation():
    """
    使用智能推薦進行分類訓練
    """
    print("🧠 智能推薦 - 分類任務")
    print("=" * 50)

    # 步驟 1: 分析資料集
    data_path = "path/to/your/classification/dataset"
    analyzer = DatasetAnalyzer()

    print("📊 分析資料集特徵...")
    stats = analyzer.analyze_classification_dataset(f"{data_path}/train")

    print(f"   類別數量: {stats['num_classes']}")
    print(f"   總圖片數: {stats['total_images']}")
    print(f"   平均圖片大小: {stats['avg_image_size']}")
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
    trainer.train()

    print("\n✅ 智能推薦訓練完成！")

def detection_with_smart_recommendation():
    """
    使用智能推薦進行物件偵測訓練
    """
    print("\n🧠 智能推薦 - 物件偵測任務")
    print("=" * 50)

    # 步驟 1: 分析YOLO資料集
    data_yaml = "path/to/your/yolo/data.yaml"
    analyzer = DatasetAnalyzer()

    print("📊 分析YOLO資料集...")
    stats = analyzer.analyze_yolo_dataset(data_yaml)

    print(f"   類別數量: {stats['num_classes']}")
    print(f"   訓練圖片數: {stats['train_images']}")
    print(f"   驗證圖片數: {stats['val_images']}")
    print(f"   平均物件數/圖: {stats['avg_objects_per_image']:.2f}")
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
    trainer.train()

    print("\n✅ 智能推薦訓練完成！")

def main():
    """
    主函數 - 展示智能推薦功能
    """
    print("🚀 iVIT 2.0 智能推薦系統範例")
    print("=" * 60)

    # 選擇任務類型
    task_type = input("選擇任務類型 (1: 分類, 2: 物件偵測): ")

    if task_type == "1":
        classification_with_smart_recommendation()
    elif task_type == "2":
        detection_with_smart_recommendation()
    else:
        print("❌ 無效的選擇")

if __name__ == "__main__":
    print("🧠 此範例展示如何使用智能推薦系統")
    print("系統會自動分析您的資料集並推薦最佳參數")
    print("請確保資料集路徑正確後執行")
    # main()  # 取消註解來執行
