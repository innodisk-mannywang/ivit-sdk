#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 分割推理範例
展示如何使用分割推理器進行語義分割
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from ivit.inference import SegmentationInference


def main():
    """主函數 - 分割推理範例"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 分割推理範例')
    parser.add_argument('--image', type=str, required=True,
                       help='輸入圖像路徑')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型檔案路徑 (如果未指定，使用預訓練模型)')
    parser.add_argument('--model_name', type=str, default='yolov8n-seg.pt',
                       help='模型名稱 (yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt)')
    parser.add_argument('--num_classes', type=int, default=21,
                       help='類別數量')
    parser.add_argument('--class_names', type=str, default=None,
                       help='類別名稱檔案路徑 (JSON格式)')
    parser.add_argument('--device', type=str, default='auto',
                       help='設備配置 (auto, cpu, cuda)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='信心度閾值')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                       help='IoU閾值')
    parser.add_argument('--output_dir', type=str, default='./segmentation_output',
                       help='輸出目錄')
    parser.add_argument('--save_masks', action='store_true',
                       help='保存遮罩圖像')
    parser.add_argument('--save_overlay', action='store_true',
                       help='保存遮罩疊加圖像')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='遮罩疊加透明度')
    
    args = parser.parse_args()
    
    print("🚀 iVIT 2.0 分割推理範例")
    print("=" * 50)
    
    # 檢查輸入圖像
    if not os.path.exists(args.image):
        print(f"❌ 圖像檔案不存在: {args.image}")
        return
    
    # 載入類別名稱 (如果提供)
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        import json
        with open(args.class_names, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"✅ 載入類別名稱: {len(class_names)} 個類別")
    
    # 創建分割推理器
    print(f"🔧 創建分割推理器...")
    print(f"   模型: {args.model_name}")
    print(f"   類別數: {args.num_classes}")
    print(f"   信心度閾值: {args.conf_threshold}")
    print(f"   設備: {args.device}")
    
    inference = SegmentationInference(
        model_name=args.model_name,
        num_classes=args.num_classes,
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
        print(f"   偵測到遮罩數: {results['num_masks']}")
        
        if results['masks']:
            print(f"\n🔍 偵測到的遮罩:")
            for i, mask_info in enumerate(results['masks']):
                bbox = mask_info['bbox']
                print(f"   {i+1}. {mask_info['class_name']}: {mask_info['confidence']:.4f}")
                print(f"      位置: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) -> ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
                print(f"      遮罩面積: {mask_info['area']:.1f} 像素")
        
        # 獲取遮罩統計
        stats = inference.get_mask_statistics(results['masks'])
        print(f"\n📊 遮罩統計:")
        print(f"   總遮罩數: {stats['total_masks']}")
        print(f"   類別分佈: {stats['class_counts']}")
        print(f"   面積統計: 平均={stats['area_stats']['mean']:.1f}, "
              f"最大={stats['area_stats']['max']:.1f}")
        print(f"   信心度統計: 平均={stats['confidence_stats']['mean']:.4f}, "
              f"最大={stats['confidence_stats']['max']:.4f}")
        
        # 創建輸出目錄
        if args.save_masks or args.save_overlay:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存遮罩圖像
        if args.save_masks and results['masks']:
            print(f"\n💾 保存遮罩圖像...")
            for i, mask_info in enumerate(results['masks']):
                mask = mask_info['mask']
                mask_path = output_dir / f"mask_{i+1}_{mask_info['class_name']}.png"
                cv2.imwrite(str(mask_path), mask * 255)
                print(f"   遮罩 {i+1} 已保存: {mask_path}")
        
        # 創建遮罩疊加圖像
        if args.save_overlay and results['masks']:
            print(f"\n🎨 創建遮罩疊加圖像...")
            overlay_image = inference.create_colored_mask(args.image, results['masks'], alpha=args.alpha)
            
            overlay_path = output_dir / f"overlay_{Path(args.image).stem}.jpg"
            cv2.imwrite(str(overlay_path), overlay_image)
            print(f"   疊加圖像已保存: {overlay_path}")
        
        # 創建類別遮罩
        if results['masks'] and results['image_shape']:
            print(f"\n🏷️ 創建類別遮罩...")
            class_mask = inference.create_class_mask(results['image_shape'], results['masks'])
            
            if args.save_masks:
                class_mask_path = output_dir / f"class_mask_{Path(args.image).stem}.png"
                cv2.imwrite(str(class_mask_path), class_mask)
                print(f"   類別遮罩已保存: {class_mask_path}")
        
        # 提取輪廓
        if results['masks']:
            print(f"\n🔍 提取輪廓...")
            contours = inference.extract_contours(results['masks'])
            print(f"   找到 {len(contours)} 個輪廓")
            
            for i, contour_info in enumerate(contours[:5]):  # 只顯示前5個
                print(f"   輪廓 {i+1}: {contour_info['class_name']}, "
                      f"面積={contour_info['area']:.1f}, "
                      f"周長={contour_info['perimeter']:.1f}")
        
        # 遮罩過濾測試
        print(f"\n🔍 遮罩過濾測試:")
        
        # 按類別過濾
        if results['masks']:
            target_classes = [0, 1, 2]  # 假設過濾前3個類別
            class_filtered = inference.filter_masks_by_class(results['masks'], target_classes)
            print(f"   類別過濾 (類別 {target_classes}): {len(class_filtered)} 個遮罩")
        
        # 按面積過濾
        if results['masks']:
            min_area = stats['area_stats']['mean']
            area_filtered = inference.filter_masks_by_area(results['masks'], min_area=min_area)
            print(f"   面積過濾 (最小面積 {min_area:.1f}): {len(area_filtered)} 個遮罩")
        
        # 合併遮罩
        if results['masks']:
            print(f"\n🔗 合併遮罩...")
            merged_mask = inference.merge_masks(results['masks'])
            if args.save_masks and merged_mask.size > 0:
                merged_path = output_dir / f"merged_mask_{Path(args.image).stem}.png"
                cv2.imwrite(str(merged_path), merged_mask * 255)
                print(f"   合併遮罩已保存: {merged_path}")
        
        # 效能測試
        print(f"\n⚡ 效能測試...")
        benchmark_results = inference.benchmark(args.image, num_runs=10)
        print(f"   平均推理時間: {benchmark_results['mean_time']:.4f} 秒")
        print(f"   推理速度: {benchmark_results['fps']:.2f} FPS")
        
    except Exception as e:
        print(f"❌ 推理失敗: {str(e)}")
        return
    
    print(f"\n✅ 推理完成！")


if __name__ == "__main__":
    main()
