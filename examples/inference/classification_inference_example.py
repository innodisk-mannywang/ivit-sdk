#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 分類推理範例
展示如何使用分類推理器進行圖像分類
"""

import os
import sys
import argparse
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from ivit.inference import ClassificationInference


def main():
    """主函數 - 分類推理範例"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 分類推理範例')
    parser.add_argument('--image', type=str, required=True,
                       help='輸入圖像路徑')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型檔案路徑 (如果未指定，使用預訓練模型)')
    parser.add_argument('--model_name', type=str, default='resnet18',
                       help='模型名稱 (resnet18, resnet50, efficientnet_b0)')
    parser.add_argument('--num_classes', type=int, default=1000,
                       help='類別數量')
    parser.add_argument('--class_names', type=str, default=None,
                       help='類別名稱檔案路徑 (JSON格式，如果未指定且使用自定義模型，會自動尋找對應的 class_names.json)')
    parser.add_argument('--device', type=str, default='auto',
                       help='設備配置 (auto, cpu, cuda)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='顯示前K個預測結果')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                       help='信心度閾值')
    
    args = parser.parse_args()
    
    print("🚀 iVIT 2.0 分類推理範例")
    print("=" * 50)
    
    # 檢查輸入圖像
    if not os.path.exists(args.image):
        print(f"❌ 圖像檔案不存在: {args.image}")
        return
    
    # 載入類別名稱 (如果提供或自動尋找)
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        import json
        with open(args.class_names, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"✅ 載入類別名稱: {len(class_names)} 個類別")
    elif args.model_path and os.path.exists(args.model_path):
        # 自動尋找對應的 class_names.json 檔案
        model_dir = os.path.dirname(args.model_path)
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        auto_class_names_path = os.path.join(model_dir, f"{model_name}_class_names.json")
        
        if os.path.exists(auto_class_names_path):
            import json
            with open(auto_class_names_path, 'r', encoding='utf-8') as f:
                class_names = json.load(f)
            print(f"✅ 自動載入類別名稱: {len(class_names)} 個類別")
            print(f"   檔案路徑: {auto_class_names_path}")
        else:
            print("ℹ️ 未找到對應的 class_names.json 檔案，將使用預設類別名稱")
    
    # 創建分類推理器
    print(f"🔧 創建分類推理器...")
    print(f"   模型: {args.model_name}")
    print(f"   類別數: {args.num_classes}")
    print(f"   設備: {args.device}")
    
    inference = ClassificationInference(
        model_name=args.model_name,
        num_classes=args.num_classes,
        class_names=class_names,
        device=args.device
    )
    
    # 載入模型 (如果提供)
    if args.model_path and os.path.exists(args.model_path):
        print(f"📁 載入自定義模型: {args.model_path}")
        inference.load_model(args.model_path)
    else:
        print("📁 使用預訓練模型")
    
    # 顯示模型資訊
    model_info = inference.get_model_info()
    print(f"\n📊 模型資訊:")
    print(f"   總參數數: {model_info['total_parameters']:,}")
    print(f"   模型大小: {model_info['model_size_mb']:.2f} MB")
    print(f"   設備: {model_info['device']}")
    
    # 執行推理
    print(f"\n🎯 執行推理...")
    print(f"   輸入圖像: {args.image}")
    
    try:
        # 基本推理
        results = inference.predict(args.image)
        
        print(f"\n📋 推理結果:")
        print(f"   預測類別: {results['top_class']}")
        print(f"   信心度: {results['top_confidence']:.4f}")
        
        print(f"\n🏆 前 {min(args.top_k, len(results['predictions']))} 個預測:")
        for i, pred in enumerate(results['predictions'][:args.top_k]):
            print(f"   {i+1}. {pred['class_name']}: {pred['confidence']:.4f}")
        
        # 信心度過濾
        if args.confidence_threshold > 0:
            print(f"\n🔍 信心度過濾 (閾值: {args.confidence_threshold}):")
            filtered_results = inference.predict_with_confidence_threshold(
                args.image, args.confidence_threshold
            )
            print(f"   符合條件的預測數: {len(filtered_results['predictions'])}")
        
        # 獲取所有類別機率
        print(f"\n📊 所有類別機率 (前10個):")
        probabilities = inference.get_class_probabilities(args.image)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, prob) in enumerate(sorted_probs[:10]):
            print(f"   {i+1}. {class_name}: {prob:.4f}")
        
        # 效能測試
        print(f"\n⚡ 效能測試...")
        benchmark_results = inference.benchmark(args.image, num_runs=50)
        print(f"   平均推理時間: {benchmark_results['mean_time']:.4f} 秒")
        print(f"   推理速度: {benchmark_results['fps']:.2f} FPS")
        
    except Exception as e:
        print(f"❌ 推理失敗: {str(e)}")
        return
    
    print(f"\n✅ 推理完成！")


if __name__ == "__main__":
    main()
