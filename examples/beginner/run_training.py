#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 統一訓練啟動腳本
支援所有訓練類型的單卡/多卡訓練
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def detect_num_classes(data_path: str, task_type: str) -> int:
    """
    自動檢測資料集的類別數量
    
    Args:
        data_path: 資料集路徑
        task_type: 任務類型
        
    Returns:
        類別數量
    """
    data_path = Path(data_path)
    
    if task_type == 'classification':
        # 對於分類任務，檢查 train 目錄下的子目錄數量
        train_path = data_path / 'train'
        if train_path.exists():
            classes = [d for d in train_path.iterdir() if d.is_dir()]
            return len(classes)
        else:
            # 如果沒有 train 目錄，檢查根目錄下的子目錄
            classes = [d for d in data_path.iterdir() if d.is_dir()]
            return len(classes)
    
    elif task_type in ['detection', 'segmentation']:
        # 對於檢測和分割任務，檢查 labels 目錄下的 .txt 文件
        labels_path = data_path / 'labels'
        if labels_path.exists():
            # 讀取一個標籤文件來確定類別數量
            label_files = list(labels_path.glob('*.txt'))
            if label_files:
                with open(label_files[0], 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # 找到最大的類別 ID
                        max_class_id = 0
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                max_class_id = max(max_class_id, class_id)
                        return max_class_id + 1
        
        # 如果沒有找到標籤文件，嘗試從 data.yaml 讀取
        yaml_path = data_path / 'data.yaml'
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'nc' in data:
                        return data['nc']
            except:
                pass
    
    # 預設值
    return 3


def detect_img_size(data_path: str, task_type: str) -> int:
    """
    自動檢測資料集的圖片尺寸
    
    Args:
        data_path: 資料集路徑
        task_type: 任務類型
        
    Returns:
        圖片尺寸
    """
    if task_type == 'classification':
        return 224  # 分類任務通常使用 224x224
    elif task_type in ['detection', 'segmentation']:
        return 640  # 檢測和分割任務通常使用 640x640
    else:
        return 224


def run_training(task_type: str, data_path: str, device: str, epochs: int = 50,
                batch_size: int = 16, learning_rate: float = 0.01, **kwargs):
    """
    執行指定類型的訓練
    
    Args:
        task_type: 訓練類型 (classification, detection, segmentation, custom)
        data_path: 資料集路徑
        device: 設備配置
        epochs: 訓練輪數
        batch_size: 批次大小
        learning_rate: 學習率
        **kwargs: 其他參數
    """
    
    # 獲取腳本目錄
    script_dir = Path(__file__).parent
    
    # 根據任務類型選擇對應的腳本
    script_map = {
        'classification': script_dir / 'classification_training.py',
        'detection': script_dir / 'detection_training.py',
        'segmentation': script_dir / 'segmentation_training.py',
        'custom': script_dir / 'custom_training.py'
    }
    
    if task_type not in script_map:
        print(f"❌ 不支援的任務類型: {task_type}")
        print(f"支援的類型: {list(script_map.keys())}")
        return False
    
    script_path = script_map[task_type]
    
    # 構建命令
    cmd = [
        sys.executable, str(script_path),
        '--data_path', data_path,
        '--device', device,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--learning_rate', str(learning_rate)
    ]
    
    # 添加額外參數
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    print(f"🚀 執行命令: {' '.join(cmd)}")
    print("=" * 80)
    
    # 執行訓練
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 訓練失敗，返回碼: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ 執行錯誤: {str(e)}")
        return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 統一訓練啟動腳本')
    
    # 基本參數
    parser.add_argument('--task', type=str, required=True,
                       choices=['classification', 'detection', 'segmentation', 'custom'],
                       help='訓練任務類型')
    parser.add_argument('--data_path', type=str, required=True,
                       help='資料集路徑')
    parser.add_argument('--device', type=str, default='0',
                       help='設備配置 (單卡: "0", 多卡: "0,1,2,3")')
    parser.add_argument('--epochs', type=int, default=50,
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='學習率')
    
    # 模型參數
    parser.add_argument('--model', type=str, default=None,
                       help='模型名稱')
    parser.add_argument('--img_size', type=int, default=None,
                       help='圖片尺寸')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='類別數量 (classification, custom)')
    
    args = parser.parse_args()
    
    # 自動檢測資料集參數
    print("🔍 自動檢測資料集參數...")
    auto_num_classes = detect_num_classes(args.data_path, args.task)
    auto_img_size = detect_img_size(args.data_path, args.task)
    
    print(f"📊 檢測到類別數量: {auto_num_classes}")
    print(f"📐 檢測到圖片尺寸: {auto_img_size}")
    
    # 準備額外參數
    extra_kwargs = {}
    if args.model:
        extra_kwargs['model'] = args.model
    if args.img_size:
        extra_kwargs['img_size'] = args.img_size
    else:
        extra_kwargs['img_size'] = auto_img_size
    if args.num_classes:
        extra_kwargs['num_classes'] = args.num_classes
    else:
        extra_kwargs['num_classes'] = auto_num_classes
    
    # 顯示配置
    print("🚀 iVIT 2.0 統一訓練啟動")
    print("=" * 50)
    print(f"📋 任務類型: {args.task}")
    print(f"📁 資料路徑: {args.data_path}")
    print(f"🔧 設備: {args.device}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"📈 學習率: {args.learning_rate}")
    if extra_kwargs:
        print("🔧 額外參數:")
        for key, value in extra_kwargs.items():
            print(f"   {key}: {value}")
    print("=" * 50)
    
    # 執行訓練
    success = run_training(
        task_type=args.task,
        data_path=args.data_path,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        **extra_kwargs
    )
    
    if success:
        print("\n✅ 訓練成功完成！")
    else:
        print("\n❌ 訓練失敗！")
        sys.exit(1)


if __name__ == "__main__":
    main()
