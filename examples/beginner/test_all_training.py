#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 測試所有訓練類型
用於驗證所有訓練腳本是否正常工作
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_training_script(script_path: str, task_type: str, data_path: str, 
                        device: str = '0', epochs: int = 1, **kwargs):
    """測試單個訓練腳本"""
    print(f"\n🧪 測試 {task_type} 訓練...")
    print("=" * 50)
    
    # 構建命令
    cmd = [sys.executable, script_path, '--data_path', data_path, '--device', device, '--epochs', str(epochs)]
    
    # 添加額外參數
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    print(f"執行命令: {' '.join(cmd)}")
    
    try:
        # 執行測試
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分鐘超時
        
        if result.returncode == 0:
            print(f"✅ {task_type} 訓練測試通過")
            return True
        else:
            print(f"❌ {task_type} 訓練測試失敗")
            print(f"錯誤輸出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {task_type} 訓練測試超時")
        return False
    except Exception as e:
        print(f"❌ {task_type} 訓練測試錯誤: {str(e)}")
        return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='測試所有訓練類型')
    parser.add_argument('--data_path', type=str, required=True,
                       help='測試資料集路徑')
    parser.add_argument('--device', type=str, default='0',
                       help='設備配置')
    parser.add_argument('--epochs', type=int, default=1,
                       help='測試輪數')
    parser.add_argument('--test_types', nargs='+', 
                       choices=['classification', 'detection', 'segmentation', 'custom'],
                       default=['classification', 'detection', 'segmentation', 'custom'],
                       help='要測試的訓練類型')
    
    args = parser.parse_args()
    
    # 獲取腳本目錄
    script_dir = Path(__file__).parent
    
    # 測試結果
    results = {}
    
    print("🚀 iVIT 2.0 SDK 訓練測試")
    print("=" * 50)
    print(f"📁 測試資料路徑: {args.data_path}")
    print(f"🔧 設備: {args.device}")
    print(f"🔄 測試輪數: {args.epochs}")
    print(f"📋 測試類型: {args.test_types}")
    
    # 測試各類型
    test_configs = {
        'classification': {
            'script': script_dir / 'classification_training.py',
            'kwargs': {'num_classes': 3, 'model': 'resnet18', 'img_size': 224}
        },
        'detection': {
            'script': script_dir / 'detection_training.py',
            'kwargs': {'model': 'yolov8n.pt', 'img_size': 640}
        },
        'segmentation': {
            'script': script_dir / 'segmentation_training.py',
            'kwargs': {'model': 'yolov8n-seg.pt', 'img_size': 640}
        },
        'custom': {
            'script': script_dir / 'custom_training.py',
            'kwargs': {'num_classes': 3, 'model': 'custom', 'img_size': 224}
        }
    }
    
    for task_type in args.test_types:
        if task_type in test_configs:
            config = test_configs[task_type]
            success = test_training_script(
                str(config['script']),
                task_type,
                args.data_path,
                args.device,
                args.epochs,
                **config['kwargs']
            )
            results[task_type] = success
        else:
            print(f"⚠️ 不支援的測試類型: {task_type}")
            results[task_type] = False
    
    # 顯示測試結果
    print("\n📊 測試結果總結")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for task_type, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"{task_type:15} : {status}")
        if success:
            passed += 1
    
    print("=" * 50)
    print(f"總計: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！")
        return 0
    else:
        print("⚠️ 部分測試失敗，請檢查配置")
        return 1


if __name__ == "__main__":
    sys.exit(main())
