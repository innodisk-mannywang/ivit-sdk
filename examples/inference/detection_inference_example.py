#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 偵測推理範例
展示如何使用偵測推理器進行物件偵測
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from ivit.inference import DetectionInference


def load_class_names(file_path: str) -> Optional[List[str]]:
    """
    載入類別名稱，支援 JSON 和 YAML 格式
    
    Args:
        file_path: 類別名稱檔案路徑
        
    Returns:
        類別名稱列表，如果載入失敗則返回 None
    """
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            # 載入 JSON 格式
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                class_names = json.load(f)
            return class_names
            
        elif file_ext in ['.yaml', '.yml']:
            # 載入 YAML 格式
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 從 YAML 中提取類別名稱
            if 'names' in data:
                if isinstance(data['names'], list):
                    return data['names']
                elif isinstance(data['names'], dict):
                    # 如果是字典格式，按索引排序
                    return [data['names'][str(i)] for i in range(len(data['names']))]
            
            print(f"⚠️ YAML 檔案中未找到 'names' 欄位")
            return None
            
        else:
            print(f"⚠️ 不支援的檔案格式: {file_ext}")
            return None
            
    except Exception as e:
        print(f"❌ 載入類別名稱失敗: {e}")
        return None


def main():
    """主函數 - 偵測推理範例"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 偵測推理範例')
    parser.add_argument('--image', type=str, required=True,
                       help='輸入圖像路徑')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型檔案路徑 (如果未指定，使用預訓練模型)')
    parser.add_argument('--model_name', type=str, default='yolov8n.pt',
                       help='模型名稱 (yolov8n.pt, yolov8s.pt, yolov8m.pt)')
    parser.add_argument('--class_names', type=str, default=None,
                       help='類別名稱檔案路徑 (支援 JSON 和 YAML 格式，如果未指定且使用自定義模型，會自動尋找對應的 class_names.json)')
    parser.add_argument('--device', type=str, default='auto',
                       help='設備配置 (auto, cpu, cuda)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='信心度閾值')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                       help='IoU閾值')
    parser.add_argument('--output_dir', type=str, default='./detection_output',
                       help='輸出目錄')
    parser.add_argument('--draw_boxes', action='store_true',
                       help='繪製偵測框')
    parser.add_argument('--save_image', action='store_true',
                       help='保存結果圖像')
    
    args = parser.parse_args()
    
    print("🚀 iVIT 2.0 偵測推理範例")
    print("=" * 50)
    
    # 檢查輸入圖像
    if not os.path.exists(args.image):
        print(f"❌ 圖像檔案不存在: {args.image}")
        return
    
    # 載入類別名稱 (如果提供或自動尋找)
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        class_names = load_class_names(args.class_names)
        if class_names:
            print(f"✅ 載入類別名稱: {len(class_names)} 個類別")
    elif args.model_path and os.path.exists(args.model_path):
        # 自動尋找對應的 class_names.json 檔案
        model_dir = os.path.dirname(args.model_path)
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        auto_class_names_path = os.path.join(model_dir, f"{model_name}_class_names.json")
        
        if os.path.exists(auto_class_names_path):
            class_names = load_class_names(auto_class_names_path)
            if class_names:
                print(f"✅ 自動載入類別名稱: {len(class_names)} 個類別")
                print(f"   檔案路徑: {auto_class_names_path}")
        
        if not class_names:
            print("ℹ️ 未找到對應的 class_names.json 檔案，將使用預設類別名稱")
    
    # 創建偵測推理器
    print(f"🔧 創建偵測推理器...")
    print(f"   模型: {args.model_name}")
    print(f"   信心度閾值: {args.conf_threshold}")
    print(f"   IoU閾值: {args.iou_threshold}")
    print(f"   設備: {args.device}")
    
    inference = DetectionInference(
        model_name=args.model_name,
        class_names=class_names,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
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
        print(f"   偵測到物件數: {results['num_detections']}")
        
        if results['detections']:
            print(f"\n🔍 偵測到的物件:")
            for i, detection in enumerate(results['detections']):
                bbox = detection['bbox']
                print(f"   {i+1}. {detection['class_name']}: {detection['confidence']:.4f}")
                print(f"      位置: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) -> ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
                print(f"      大小: {bbox['width']:.1f} x {bbox['height']:.1f}")
                print(f"      面積: {detection['area']:.1f}")
        
        # 獲取偵測統計
        stats = inference.get_detection_statistics(results)
        print(f"\n📊 偵測統計:")
        print(f"   總偵測數: {stats['total_detections']}")
        print(f"   類別分佈: {stats['class_counts']}")
        
        if stats['total_detections'] > 0:
            print(f"   信心度統計: 平均={stats['confidence_stats']['mean']:.4f}, "
                  f"最大={stats['confidence_stats']['max']:.4f}")
            print(f"   大小統計: 平均面積={stats['size_stats']['mean_area']:.1f}, "
                  f"最大面積={stats['size_stats']['max_area']:.1f}")
        else:
            print("   無偵測結果，無法計算統計資訊")
        
        # 繪製偵測框
        if args.draw_boxes or args.save_image:
            print(f"\n🎨 繪製偵測框...")
            output_image = inference.draw_detections(args.image, results)
            
            if args.save_image:
                # 創建輸出目錄
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存結果圖像
                output_path = output_dir / f"detection_result_{Path(args.image).stem}.jpg"
                cv2.imwrite(str(output_path), output_image)
                print(f"💾 結果圖像已保存: {output_path}")
        
        # 信心度過濾測試
        print(f"\n🔍 信心度過濾測試:")
        for threshold in [0.5, 0.7, 0.9]:
            filtered_results = inference.filter_by_confidence(results, threshold)
            print(f"   閾值 {threshold}: {filtered_results['num_detections']} 個偵測")
        
        # 大小過濾測試
        print(f"\n📏 大小過濾測試:")
        if stats['total_detections'] > 0:
            min_area = stats['size_stats']['mean_area']
            size_filtered = inference.filter_by_size(results, min_area=min_area)
            print(f"   最小面積 {min_area:.1f}: {size_filtered['num_detections']} 個偵測")
        else:
            print("   無偵測結果，跳過大小過濾測試")
        
        # 效能測試
        print(f"\n⚡ 效能測試...")
        benchmark_results = inference.benchmark(args.image, num_runs=20)
        print(f"   平均推理時間: {benchmark_results['mean_time']:.4f} 秒")
        print(f"   推理速度: {benchmark_results['fps']:.2f} FPS")
        
    except Exception as e:
        print(f"❌ 推理失敗: {str(e)}")
        return
    
    print(f"\n✅ 推理完成！")


if __name__ == "__main__":
    main()
