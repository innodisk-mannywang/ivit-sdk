#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 統一訓練指南
支援 Classification, Detection, Segmentation 的單卡/多卡訓練
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ivit.trainer.classification import ClassificationTrainer
from ivit.trainer.detection import DetectionTrainer
from ivit.trainer.segmentation import SegmentationTrainer


class UnifiedTrainingGuide:
    """統一訓練指南類別"""
    
    def __init__(self):
        self.trainers = {
            'classification': ClassificationTrainer,
            'detection': DetectionTrainer,
            'segmentation': SegmentationTrainer
        }
    
    def get_training_config(self, task_type: str, device: str, epochs: int = 50) -> Dict[str, Any]:
        """獲取不同任務的訓練配置"""
        
        base_config = {
            'epochs': epochs,
            'device': device,
            'batch_size': 16,
            'learning_rate': 0.01
        }
        
        if task_type == 'classification':
            return {
                **base_config,
                'model_name': 'resnet18',
                'img_size': 224,
                'num_classes': 3,  # 根據資料集調整
                'data_path': '/path/to/classification/dataset'
            }
        
        elif task_type == 'detection':
            return {
                **base_config,
                'model_name': 'yolov8n.pt',
                'img_size': 640,
                'data_path': '/path/to/yolo/dataset'
            }
        
        elif task_type == 'segmentation':
            return {
                **base_config,
                'model_name': 'yolov8n-seg.pt',
                'img_size': 640,
                'data_path': '/path/to/yolo/dataset'
            }
        
        else:
            raise ValueError(f"不支援的任務類型: {task_type}")
    
    def create_trainer(self, task_type: str, config: Dict[str, Any]):
        """創建對應的訓練器"""
        if task_type not in self.trainers:
            raise ValueError(f"不支援的任務類型: {task_type}")
        
        trainer_class = self.trainers[task_type]
        
        if task_type == 'classification':
            return trainer_class(
                model_name=config['model_name'],
                img_size=config['img_size'],
                num_classes=config['num_classes'],
                learning_rate=config['learning_rate'],
                device=config['device']
            )
        
        elif task_type in ['detection', 'segmentation']:
            return trainer_class(
                model_name=config['model_name'],
                img_size=config['img_size'],
                learning_rate=config['learning_rate'],
                device=config['device']
            )
    
    def run_training(self, task_type: str, config: Dict[str, Any], data_path: str):
        """執行訓練"""
        print(f"🚀 開始 {task_type} 訓練...")
        print(f"📁 資料路徑: {data_path}")
        print(f"🔧 設備: {config['device']}")
        print(f"📊 Epochs: {config['epochs']}")
        
        # 創建訓練器
        trainer = self.create_trainer(task_type, config)
        
        # 執行訓練
        if task_type == 'classification':
            results = trainer.train(
                dataset_path=data_path,
                epochs=config['epochs'],
                batch_size=config['batch_size']
            )
        else:  # detection, segmentation
            results = trainer.train(
                dataset_path=data_path,
                epochs=config['epochs'],
                batch_size=config['batch_size']
            )
        
        print("🎉 訓練完成！")
        return results


def main():
    """主函數 - 命令行介面"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 統一訓練指南')
    parser.add_argument('--task', type=str, required=True, 
                       choices=['classification', 'detection', 'segmentation'],
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
    
    args = parser.parse_args()
    
    # 創建訓練指南
    guide = UnifiedTrainingGuide()
    
    # 獲取配置
    config = guide.get_training_config(args.task, args.device, args.epochs)
    config['data_path'] = args.data_path
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    
    # 執行訓練
    try:
        results = guide.run_training(args.task, config, args.data_path)
        print(f"📊 最終結果: {results}")
    except Exception as e:
        print(f"❌ 訓練失敗: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
