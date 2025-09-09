#!/usr/bin/env python3
"""
iVIT 2.0 SDK - 專家範例：自定義訓練流程
這個範例展示如何完全自定義訓練流程，包括自定義損失函數、學習率調度等
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from ivit.trainer.classification import ClassificationConfig, ClassificationTrainer
from ivit.core.base_trainer import BaseTrainer
import logging

class CustomLoss(nn.Module):
    """
    自定義損失函數：結合交叉熵和焦點損失
    """
    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss

class ExpertClassificationTrainer(ClassificationTrainer):
    """
    專家級自定義分類訓練器
    """

    def __init__(self, config, custom_loss=None, custom_scheduler=None):
        super().__init__(config)
        self.custom_loss = custom_loss
        self.custom_scheduler = custom_scheduler

        # 設置詳細日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_training_components(self):
        """
        覆寫父類方法，使用自定義組件
        """
        super().setup_training_components()

        # 使用自定義損失函數
        if self.custom_loss:
            self.criterion = self.custom_loss
            self.logger.info("使用自定義損失函數")

        # 使用自定義學習率調度器
        if self.custom_scheduler:
            self.scheduler = self.custom_scheduler(self.optimizer)
            self.logger.info(f"使用自定義學習率調度器: {type(self.scheduler).__name__}")

    def train_epoch(self, epoch):
        """
        自定義單個epoch的訓練過程
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # 添加進度條和詳細日誌
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 統計
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # 更新進度條
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # 每100個批次記錄一次
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Epoch: {epoch+1}, Batch: {batch_idx}, '
                    f'Loss: {loss.item():.6f}, '
                    f'Acc: {100.*correct/total:.2f}%'
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate_epoch(self):
        """
        自定義驗證過程，添加更多指標
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # 每類別準確率統計
                for i in range(target.size(0)):
                    label = target[i].item()
                    pred = predicted[i].item()

                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0

                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        # 記錄每類別準確率
        self.logger.info("每類別準確率:")
        for class_idx in sorted(class_total.keys()):
            class_acc = 100. * class_correct[class_idx] / class_total[class_idx]
            self.logger.info(f"  Class {class_idx}: {class_acc:.2f}%")

        return avg_loss, accuracy

def expert_training_example():
    """
    專家級訓練範例
    """
    print("🎓 iVIT 2.0 專家級自定義訓練")
    print("=" * 50)

    # 步驟 1: 設置基本配置
    config = ClassificationConfig(
        train_data="path/to/train/data",
        val_data="path/to/val/data",
        num_classes=10,
        model_name="resnet50",
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        output_dir="./expert_output"
    )

    # 步驟 2: 創建自定義損失函數
    # 計算類別權重（處理類別不平衡）
    class_weights = torch.tensor([1.0, 2.0, 1.5, 1.0, 3.0, 1.0, 1.2, 1.0, 2.5, 1.0])
    custom_loss = CustomLoss(alpha=0.25, gamma=2.0, class_weights=class_weights)

    print("✅ 創建自定義焦點損失函數（處理類別不平衡）")

    # 步驟 3: 創建自定義學習率調度器
    def custom_scheduler_fn(optimizer):
        # 組合調度器：前50輪使用StepLR，後50輪使用CosineAnnealing
        return StepLR(optimizer, step_size=30, gamma=0.1)

    print("✅ 創建自定義學習率調度器")

    # 步驟 4: 創建專家訓練器
    trainer = ExpertClassificationTrainer(
        config=config,
        custom_loss=custom_loss,
        custom_scheduler=custom_scheduler_fn
    )

    print("✅ 創建專家級訓練器")

    # 步驟 5: 執行訓練
    print("\n🚀 開始專家級自定義訓練...")
    trainer.train()

    print("\n🎉 專家級訓練完成！")
    print("訓練日誌已保存，包含詳細的每類別準確率統計")

def advanced_model_ensemble():
    """
    模型集成範例
    """
    print("\n🎯 模型集成範例")
    print("=" * 30)

    # 訓練多個不同的模型
    models = ["resnet18", "efficientnet-b0", "mobilenet_v3_small"]

    for model_name in models:
        print(f"\n訓練 {model_name}...")

        config = ClassificationConfig(
            train_data="path/to/train/data",
            val_data="path/to/val/data",
            num_classes=10,
            model_name=model_name,
            learning_rate=0.001,
            batch_size=32,
            epochs=50,
            output_dir=f"./ensemble_{model_name}"
        )

        trainer = ClassificationTrainer(config)
        trainer.train()

    print("\n✅ 模型集成訓練完成！")
    print("可以使用投票或加權平均來組合預測結果")

def main():
    """
    主函數
    """
    print("🎓 iVIT 2.0 專家級功能展示")
    print("=" * 60)

    choice = input("選擇範例 (1: 自定義訓練, 2: 模型集成): ")

    if choice == "1":
        expert_training_example()
    elif choice == "2":
        advanced_model_ensemble()
    else:
        print("❌ 無效選擇")

if __name__ == "__main__":
    print("🎓 此範例展示專家級功能：")
    print("   - 自定義損失函數")
    print("   - 自定義學習率調度")
    print("   - 詳細訓練日誌")
    print("   - 每類別準確率統計")
    print("   - 模型集成")
    print("請確保資料集路徑正確後執行")
    # main()  # 取消註解來執行
