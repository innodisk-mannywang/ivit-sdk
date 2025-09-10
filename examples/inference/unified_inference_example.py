#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 統一推理範例
展示如何使用統一推理接口進行多任務推理
"""

import os
import sys
import argparse
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from ivit.inference import UnifiedInference


def main():
    """主函數 - 統一推理範例"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 統一推理範例')
    parser.add_argument('--image', type=str, required=True,
                       help='輸入圖像路徑')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['classification', 'detection', 'segmentation'],
                       default=['classification'],
                       help='要執行的任務類型')
    parser.add_argument('--model_paths', type=str, nargs='+', default=None,
                       help='模型檔案路徑 (按順序對應任務)')
    parser.add_argument('--class_names', type=str, default=None,
                       help='類別名稱檔案路徑 (JSON格式)')
    parser.add_argument('--device', type=str, default='auto',
                       help='設備配置 (auto, cpu, cuda)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='信心度閾值 (偵測/分割)')
    parser.add_argument('--output_dir', type=str, default='./unified_output',
                       help='輸出目錄')
    parser.add_argument('--save_results', action='store_true',
                       help='保存結果到檔案')
    parser.add_argument('--benchmark', action='store_true',
                       help='執行效能測試')
    
    args = parser.parse_args()
    
    print("🚀 iVIT 2.0 統一推理範例")
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
    
    # 創建統一推理器
    print(f"🔧 創建統一推理器...")
    print(f"   任務類型: {args.models}")
    print(f"   設備: {args.device}")
    
    inference = UnifiedInference(device=args.device)
    
    # 載入模型
    model_paths = args.model_paths or [None] * len(args.models)
    
    for i, (task_type, model_path) in enumerate(zip(args.models, model_paths)):
        print(f"\n📁 載入 {task_type} 模型...")
        
        if model_path and os.path.exists(model_path):
            inference.load_model(task_type, model_path, class_names=class_names)
        else:
            # 使用預設模型
            if task_type == 'classification':
                inference.load_model(task_type, None, 
                                   model_name='resnet18', 
                                   num_classes=1000,
                                   class_names=class_names)
            elif task_type == 'detection':
                inference.load_model(task_type, None,
                                   model_name='yolov8n.pt',
                                   class_names=class_names)
            elif task_type == 'segmentation':
                inference.load_model(task_type, None,
                                   model_name='yolov8n-seg.pt',
                                   num_classes=21,
                                   class_names=class_names)
    
    # 顯示載入的模型
    print(f"\n📋 已載入的模型:")
    loaded_models = inference.list_loaded_models()
    for model in loaded_models:
        print(f"   ✅ {model}")
    
    # 顯示模型資訊
    model_info = inference.get_model_info()
    print(f"\n📊 模型資訊:")
    for task_type, info in model_info.items():
        if 'error' not in info:
            print(f"   {task_type}:")
            print(f"     參數數: {info['total_parameters']:,}")
            print(f"     模型大小: {info['model_size_mb']:.2f} MB")
            print(f"     設備: {info['device']}")
    
    # 執行推理
    print(f"\n🎯 執行推理...")
    print(f"   輸入圖像: {args.image}")
    
    try:
        # 單任務推理
        if len(args.models) == 1:
            task_type = args.models[0]
            print(f"\n🔍 執行 {task_type} 推理...")
            
            if task_type == 'classification':
                results = inference.predict_classification(args.image)
                print(f"   預測類別: {results['top_class']}")
                print(f"   信心度: {results['top_confidence']:.4f}")
                
            elif task_type == 'detection':
                results = inference.predict_detection(args.image, 
                                                    conf_threshold=args.conf_threshold)
                print(f"   偵測到物件數: {results['num_detections']}")
                for i, det in enumerate(results['detections'][:5]):
                    print(f"     {i+1}. {det['class_name']}: {det['confidence']:.4f}")
                    
            elif task_type == 'segmentation':
                results = inference.predict_segmentation(args.image,
                                                       conf_threshold=args.conf_threshold)
                print(f"   偵測到遮罩數: {results['num_masks']}")
                for i, mask in enumerate(results['masks'][:5]):
                    print(f"     {i+1}. {mask['class_name']}: {mask['confidence']:.4f}")
        
        # 多任務推理
        else:
            print(f"\n🔍 執行多任務推理...")
            results = inference.predict_multi_task(args.image, args.models)
            
            for task_type, result in results.items():
                print(f"\n   {task_type} 結果:")
                if 'error' in result:
                    print(f"     ❌ 錯誤: {result['error']}")
                else:
                    if task_type == 'classification':
                        print(f"     預測類別: {result['top_class']}")
                        print(f"     信心度: {result['top_confidence']:.4f}")
                    elif task_type == 'detection':
                        print(f"     偵測到物件數: {result['num_detections']}")
                    elif task_type == 'segmentation':
                        print(f"     偵測到遮罩數: {result['num_masks']}")
        
        # 效能測試
        if args.benchmark:
            print(f"\n⚡ 效能測試...")
            benchmark_results = inference.benchmark_all_models(args.image, num_runs=20)
            
            for task_type, benchmark in benchmark_results.items():
                if 'error' not in benchmark:
                    print(f"   {task_type}:")
                    print(f"     平均推理時間: {benchmark['mean_time']:.4f} 秒")
                    print(f"     推理速度: {benchmark['fps']:.2f} FPS")
                else:
                    print(f"   {task_type}: ❌ {benchmark['error']}")
        
        # 保存結果
        if args.save_results:
            print(f"\n💾 保存結果...")
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存推理結果
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"inference_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"   結果已保存: {results_file}")
        
    except Exception as e:
        print(f"❌ 推理失敗: {str(e)}")
        return
    
    print(f"\n✅ 推理完成！")


if __name__ == "__main__":
    main()
