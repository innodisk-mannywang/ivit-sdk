#!/usr/bin/env python3
"""
iVIT 2.0 SDK - Custom 訓練範例
支援自定義模型和訓練流程的單卡/多卡訓練
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ivit.core.base_trainer import BaseTrainer, TaskConfig


class CustomModel(nn.Module):
    """自定義模型範例"""
    
    def __init__(self, num_classes: int = 3, input_size: int = 224):
        super(CustomModel, self).__init__()
        
        # 簡單的 CNN 架構
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CustomDataset(Dataset):
    """自定義資料集範例"""
    
    def __init__(self, data_path: str, split: str = 'train', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        # 假設資料集結構為 ImageFolder 格式
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        # 掃描資料集
        self._scan_dataset()
    
    def _scan_dataset(self):
        """掃描資料集結構"""
        split_path = os.path.join(self.data_path, self.split)
        if not os.path.exists(split_path):
            raise ValueError(f"資料集分割路徑不存在: {split_path}")
        
        # 獲取類別
        self.classes = sorted(os.listdir(split_path))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 獲取樣本
        for class_name in self.classes:
            class_dir = os.path.join(split_path, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((
                            os.path.join(class_dir, filename),
                            self.class_to_idx[class_name]
                        ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        
        # 載入圖片
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


class CustomTrainer(BaseTrainer):
    """自定義訓練器"""
    
    def __init__(self, model_name: str = 'custom', img_size: int = 224, 
                 num_classes: int = 3, learning_rate: float = 0.01, 
                 device: str = 'cuda'):
        
        # 創建任務配置
        config = TaskConfig(
            model_name=model_name,
            img_size=img_size,
            learning_rate=learning_rate,
            device=device
        )
        
        super().__init__(config)
        
        self.num_classes = num_classes
        self.model_name = model_name
    
    def create_model(self):
        """創建自定義模型"""
        model = CustomModel(num_classes=self.num_classes, input_size=self.img_size)
        return model
    
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, 
                      split: str = 'train', shuffle: bool = True):
        """獲取資料載入器"""
        
        # 定義資料轉換
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # 創建資料集
        dataset = CustomDataset(dataset_path, split=split, transform=transform)
        
        # 創建資料載入器
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        
        return dataloader
    
    def train(self, dataset_path: str, epochs: int = 50, batch_size: int = 16):
        """執行訓練"""
        print(f"🚀 開始自定義模型訓練...")
        print(f"📁 資料路徑: {dataset_path}")
        print(f"🔧 設備: {self.device}")
        print(f"📊 模型: {self.model_name}")
        print(f"📐 圖片尺寸: {self.img_size}")
        print(f"🏷️ 類別數量: {self.num_classes}")
        print(f"🔄 Epochs: {epochs}")
        print(f"📦 批次大小: {batch_size}")
        
        # 創建模型
        self.model = self.create_model()
        self.model = self.model.to(self.device)
        
        # 設置優化器和損失函數
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # 獲取資料載入器
        train_loader = self.get_dataloader(dataset_path, batch_size, 'train', True)
        val_loader = self.get_dataloader(dataset_path, batch_size, 'val', False)
        
        # 執行訓練循環
        best_accuracy = 0.0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # 訓練階段
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 驗證階段
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = 100.0 * correct / total
            avg_loss = train_loss / len(train_loader)
            
            train_losses.append(avg_loss)
            val_accuracies.append(accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # 這裡可以添加模型保存邏輯
        
        print(f"\n🎉 訓練完成！最佳準確率: {best_accuracy:.2f}%")
        
        return {
            'final_metrics': {
                'accuracy': best_accuracy,
                'final_loss': train_losses[-1] if train_losses else 0.0
            },
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }


def run_custom_training(data_path: str, device: str, epochs: int = 50,
                       batch_size: int = 16, learning_rate: float = 0.01,
                       model_name: str = 'custom', img_size: int = 224,
                       num_classes: int = 3):
    """執行自定義訓練"""
    
    print("🚀 iVIT 2.0 Custom 訓練")
    print("=" * 50)
    print(f"📁 資料路徑: {data_path}")
    print(f"🔧 設備: {device}")
    print(f"📊 模型: {model_name}")
    print(f"📐 圖片尺寸: {img_size}")
    print(f"🏷️ 類別數量: {num_classes}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 批次大小: {batch_size}")
    print(f"📈 學習率: {learning_rate}")
    print("=" * 50)
    
    # 創建訓練器
    trainer = CustomTrainer(
        model_name=model_name,
        img_size=img_size,
        num_classes=num_classes,
        learning_rate=learning_rate,
        device=device
    )
    
    # 驗證資料集
    print("\n📋 驗證資料集格式...")
    if not os.path.exists(data_path):
        print(f"❌ 資料集路徑不存在: {data_path}")
        return False
    
    # 檢查是否有 train 和 val 目錄
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("❌ 資料集格式不正確，請確保有 train 和 val 目錄")
        print("預期結構:")
        print("dataset/")
        print("├── train/")
        print("│   ├── class1/")
        print("│   └── class2/")
        print("└── val/")
        print("    ├── class1/")
        print("    └── class2/")
        return False
    
    print("✅ 資料集格式正確")
    
    # 開始訓練
    print("\n🎯 開始訓練...")
    try:
        results = trainer.train(
            dataset_path=data_path,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("\n🎉 訓練完成！")
        print("📊 最終結果:")
        if 'final_metrics' in results:
            for metric, value in results['final_metrics'].items():
                print(f"   {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 訓練失敗: {str(e)}")
        return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 Custom 訓練')
    parser.add_argument('--data_path', type=str, required=True,
                       help='資料集路徑 (ImageFolder 格式)')
    parser.add_argument('--device', type=str, default='0',
                       help='設備配置 (單卡: "0", 多卡: "0,1,2,3")')
    parser.add_argument('--epochs', type=int, default=50,
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='學習率')
    parser.add_argument('--model', type=str, default='custom',
                       help='模型名稱')
    parser.add_argument('--img_size', type=int, default=224,
                       help='圖片尺寸')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='類別數量')
    
    args = parser.parse_args()
    
    # 執行訓練
    success = run_custom_training(
        data_path=args.data_path,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model,
        img_size=args.img_size,
        num_classes=args.num_classes
    )
    
    if success:
        print("\n✅ 訓練成功完成！")
    else:
        print("\n❌ 訓練失敗！")
        sys.exit(1)


if __name__ == "__main__":
    main()
