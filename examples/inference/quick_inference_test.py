#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 快速推理測試
用於快速測試推理功能是否正常工作
"""

import os
import sys
import argparse
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from ivit.inference import ClassificationInference, DetectionInference, SegmentationInference, UnifiedInference


def test_classification():
    """測試分類推理"""
    print("🧪 測試分類推理...")
    
    try:
        # 創建分類推理器
        inference = ClassificationInference(
            model_name='resnet18',
            num_classes=1000,
            device='cpu'  # 使用 CPU 避免 CUDA 問題
        )
        
        # 創建測試圖像 (隨機圖像)
        import numpy as np
        from PIL import Image
        
        # 創建隨機測試圖像
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        
        # 執行推理
        results = inference.predict(test_image)
        
        print(f"   ✅ 分類推理成功")
        print(f"   預測類別: {results['top_class']}")
        print(f"   信心度: {results['top_confidence']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 分類推理失敗: {str(e)}")
        return False


def test_detection():
    """測試偵測推理"""
    print("🧪 測試偵測推理...")
    
    try:
        # 創建偵測推理器
        inference = DetectionInference(
            model_name='yolov8n.pt',
            device='cpu'  # 使用 CPU 避免 CUDA 問題
        )
        
        # 創建測試圖像
        import numpy as np
        from PIL import Image
        
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        
        # 執行推理
        results = inference.predict(test_image)
        
        print(f"   ✅ 偵測推理成功")
        print(f"   偵測到物件數: {results['num_detections']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 偵測推理失敗: {str(e)}")
        return False


def test_segmentation():
    """測試分割推理"""
    print("🧪 測試分割推理...")
    
    try:
        # 創建分割推理器
        inference = SegmentationInference(
            model_name='yolov8n-seg.pt',
            num_classes=21,
            device='cpu'  # 使用 CPU 避免 CUDA 問題
        )
        
        # 創建測試圖像
        import numpy as np
        from PIL import Image
        
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        
        # 執行推理
        results = inference.predict(test_image)
        
        print(f"   ✅ 分割推理成功")
        print(f"   偵測到遮罩數: {results['num_masks']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 分割推理失敗: {str(e)}")
        return False


def test_unified():
    """測試統一推理"""
    print("🧪 測試統一推理...")
    
    try:
        # 創建統一推理器
        inference = UnifiedInference(device='cpu')
        
        # 載入分類模型
        inference.load_model('classification', None, 
                           model_name='resnet18', 
                           num_classes=1000)
        
        # 創建測試圖像
        import numpy as np
        from PIL import Image
        
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        
        # 執行推理
        results = inference.predict_classification(test_image)
        
        print(f"   ✅ 統一推理成功")
        print(f"   預測類別: {results['top_class']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 統一推理失敗: {str(e)}")
        return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 快速推理測試')
    parser.add_argument('--test_types', nargs='+', 
                       choices=['classification', 'detection', 'segmentation', 'unified'],
                       default=['classification', 'detection', 'segmentation', 'unified'],
                       help='要測試的推理類型')
    parser.add_argument('--verbose', action='store_true',
                       help='顯示詳細輸出')
    
    args = parser.parse_args()
    
    print("🚀 iVIT 2.0 快速推理測試")
    print("=" * 50)
    
    # 測試結果
    results = {}
    
    # 執行測試
    test_functions = {
        'classification': test_classification,
        'detection': test_detection,
        'segmentation': test_segmentation,
        'unified': test_unified
    }
    
    for test_type in args.test_types:
        if test_type in test_functions:
            success = test_functions[test_type]()
            results[test_type] = success
        else:
            print(f"⚠️ 不支援的測試類型: {test_type}")
            results[test_type] = False
    
    # 顯示測試結果
    print("\n📊 測試結果總結")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_type, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"{test_type:15} : {status}")
        if success:
            passed += 1
    
    print("=" * 50)
    print(f"總計: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！推理模組工作正常")
        return 0
    else:
        print("⚠️ 部分測試失敗，請檢查配置和依賴項")
        return 1


if __name__ == "__main__":
    sys.exit(main())
