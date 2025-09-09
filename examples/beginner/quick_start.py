#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 快速開始範例
展示如何使用統一的訓練腳本
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def run_quick_examples():
    """執行快速開始範例"""
    
    print("🚀 iVIT 2.0 SDK 快速開始範例")
    print("=" * 60)
    
    # 獲取腳本目錄
    script_dir = Path(__file__).parent
    
    # 範例配置
    examples = [
        {
            'name': '分類訓練 (單卡)',
            'cmd': [
                sys.executable, str(script_dir / 'run_training.py'),
                '--task', 'classification',
                '--data_path', '/path/to/classification/dataset',
                '--device', '0',
                '--epochs', '5',
                '--model', 'resnet18',
                '--num_classes', '3'
            ]
        },
        {
            'name': '偵測訓練 (多卡)',
            'cmd': [
                sys.executable, str(script_dir / 'run_training.py'),
                '--task', 'detection',
                '--data_path', '/path/to/yolo/dataset',
                '--device', '0,1',
                '--epochs', '5',
                '--model', 'yolov8n.pt'
            ]
        },
        {
            'name': '分割訓練 (單卡)',
            'cmd': [
                sys.executable, str(script_dir / 'run_training.py'),
                '--task', 'segmentation',
                '--data_path', '/path/to/yolo/dataset',
                '--device', '0',
                '--epochs', '5',
                '--model', 'yolov8n-seg.pt'
            ]
        },
        {
            'name': '自定義訓練 (單卡)',
            'cmd': [
                sys.executable, str(script_dir / 'run_training.py'),
                '--task', 'custom',
                '--data_path', '/path/to/dataset',
                '--device', '0',
                '--epochs', '5',
                '--num_classes', '3'
            ]
        }
    ]
    
    print("📋 可用的快速開始範例:")
    print()
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   命令: {' '.join(example['cmd'])}")
        print()
    
    print("💡 使用說明:")
    print("1. 將 '/path/to/...' 替換為實際的資料集路徑")
    print("2. 根據需要調整設備配置 (單卡: '0', 多卡: '0,1,2,3')")
    print("3. 調整訓練參數 (epochs, batch_size, learning_rate 等)")
    print("4. 執行命令開始訓練")
    print()
    
    print("🔧 設備配置範例:")
    print("   單卡: --device 0")
    print("   多卡: --device 0,1,2,3")
    print("   CPU:  --device cpu")
    print()
    
    print("📊 資料集格式:")
    print("   分類/自定義: ImageFolder 格式")
    print("   偵測/分割: YOLO 格式")
    print()
    
    print("📚 更多資訊:")
    print("   - 詳細文檔: examples/beginner/README_TRAINING.md")
    print("   - 測試腳本: python test_all_training.py --help")
    print("   - 統一腳本: python run_training.py --help")


def show_help():
    """顯示幫助信息"""
    print("iVIT 2.0 SDK 快速開始")
    print()
    print("用法:")
    print("  python quick_start.py          # 顯示範例")
    print("  python quick_start.py --help   # 顯示此幫助")
    print()
    print("範例:")
    print("  # 分類訓練")
    print("  python run_training.py --task classification --data_path /path/to/dataset --device 0")
    print()
    print("  # 偵測訓練 (多卡)")
    print("  python run_training.py --task detection --data_path /path/to/dataset --device 0,1")
    print()
    print("  # 分割訓練")
    print("  python run_training.py --task segmentation --data_path /path/to/dataset --device 0")
    print()
    print("  # 自定義訓練")
    print("  python run_training.py --task custom --data_path /path/to/dataset --device 0")


def main():
    """主函數"""
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_help()
    else:
        run_quick_examples()


if __name__ == "__main__":
    main()
