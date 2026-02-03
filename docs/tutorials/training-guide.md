# iVIT-SDK 訓練教學

本教學介紹如何使用 iVIT-SDK 進行遷移式學習（Transfer Learning），將預訓練模型微調到您的自訂資料集。

## 目錄

- [快速開始](#快速開始)
- [安裝訓練依賴](#安裝訓練依賴)
- [資料集準備](#資料集準備)
- [Trainer 使用指南](#trainer-使用指南)
- [Callbacks 回調函式](#callbacks-回調函式)
- [模型匯出](#模型匯出)
- [進階設定](#進階設定)
- [常見問題](#常見問題)

---

## 快速開始

```python
from ivit.train import Trainer, ImageFolderDataset

# 1. 載入資料集
dataset = ImageFolderDataset("./my_dataset", train_split=0.8)

# 2. 建立 Trainer
trainer = Trainer(
    model="resnet50",
    dataset=dataset,
    epochs=20,
    learning_rate=0.001,
    device="cuda:0",  # 或 "cpu"
)

# 3. 開始訓練
trainer.fit()

# 4. 評估模型
metrics = trainer.evaluate()
print(f"Accuracy: {metrics['accuracy']:.2%}")

# 5. 匯出模型
trainer.export("my_model.onnx")
```

---

## 安裝訓練依賴

訓練功能需要 PyTorch 和 torchvision。根據您的硬體環境選擇安裝方式：

### CPU 版本（CI/測試環境）

```bash
pip install ivit-sdk[train]
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDA GPU 版本（訓練環境）

```bash
pip install ivit-sdk[train]

# 根據系統 CUDA 版本選擇對應的 PyTorch
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 驗證安裝

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
```

---

## 資料集準備

iVIT-SDK 支援三種常見的資料集格式。

### ImageFolder 格式（推薦）

最簡單的格式，適合圖像分類任務。

```
my_dataset/
├── class_1/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── class_2/
│   ├── image_001.jpg
│   └── ...
└── class_3/
    └── ...
```

```python
from ivit.train import ImageFolderDataset

# 自動分割訓練/驗證集
train_dataset = ImageFolderDataset(
    root="./my_dataset",
    train_split=0.8,  # 80% 訓練，20% 驗證
    split="train",
    seed=42,  # 固定隨機種子，確保可重現
)

val_dataset = ImageFolderDataset(
    root="./my_dataset",
    train_split=0.8,
    split="val",
    seed=42,
)

print(f"訓練集: {len(train_dataset)} 張")
print(f"驗證集: {len(val_dataset)} 張")
print(f"類別: {train_dataset.class_names}")
```

### COCO 格式

適合物件偵測任務。

```
coco_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

```python
from ivit.train import COCODataset

train_dataset = COCODataset(
    root="./coco_dataset/images/train",
    annotation_file="./coco_dataset/annotations/instances_train.json",
)
```

### YOLO 格式

Ultralytics YOLO 格式。

```
yolo_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img_001.txt
│   │   └── ...
│   └── val/
│       └── ...
└── classes.txt
```

```python
from ivit.train import YOLODataset

train_dataset = YOLODataset(
    root="./yolo_dataset",
    split="train",
)
```

---

## Trainer 使用指南

### 支援的預訓練模型

| 模型系列 | 可用名稱 | 參數量 | 推薦用途 |
|----------|----------|--------|----------|
| ResNet | resnet18, resnet34, resnet50, resnet101 | 11M-44M | 通用分類 |
| EfficientNet | efficientnet_b0, efficientnet_b1, efficientnet_b2 | 5M-9M | 效率優先 |
| MobileNet | mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large | 2M-5M | 邊緣部署 |
| VGG | vgg16, vgg19 | 138M-143M | 精度優先 |
| DenseNet | densenet121 | 8M | 特徵重用 |

### 基本參數

```python
trainer = Trainer(
    model="resnet50",           # 模型名稱或自訂模型
    dataset=train_dataset,       # 訓練資料集
    val_dataset=val_dataset,     # 驗證資料集（選用）
    epochs=20,                   # 訓練輪數
    learning_rate=0.001,         # 學習率
    batch_size=32,               # 批次大小
    optimizer="adam",            # 優化器：adam, adamw, sgd
    device="cuda:0",             # 訓練裝置
    freeze_backbone=True,        # 凍結骨幹網路（遷移式學習）
    num_workers=4,               # 資料載入執行緒數
)
```

### 遷移式學習設定

#### 骨幹網路凍結（推薦小資料集）

```python
# 只訓練分類頭，凍結骨幹網路
trainer = Trainer(
    model="resnet50",
    dataset=dataset,
    freeze_backbone=True,  # 凍結骨幹
    learning_rate=0.001,   # 可使用較高學習率
)
```

#### 完整微調（推薦大資料集）

```python
# 訓練整個網路
trainer = Trainer(
    model="resnet50",
    dataset=dataset,
    freeze_backbone=False,  # 訓練所有參數
    learning_rate=0.0001,   # 使用較低學習率
)
```

### 訓練歷程

```python
history = trainer.fit()

# history 是一個 list，每個元素是一個 epoch 的指標
for epoch, metrics in enumerate(history):
    print(f"Epoch {epoch + 1}:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    if 'val_loss' in metrics:
        print(f"  Val Loss: {metrics['val_loss']:.4f}")
        print(f"  Val Accuracy: {metrics['val_accuracy']:.2f}%")
```

---

## Callbacks 回調函式

Callbacks 讓您在訓練過程中執行自訂操作。

### EarlyStopping - 早停

當驗證指標不再改善時自動停止訓練。

```python
from ivit.train import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",  # 監控指標
    patience=5,          # 連續 5 個 epoch 無改善則停止
    min_delta=0.001,     # 最小改善幅度
    mode="min",          # "min" 表示越小越好
)

trainer.fit(callbacks=[early_stop])
```

### ModelCheckpoint - 模型存檔

自動儲存最佳模型。

```python
from ivit.train import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath="checkpoints/best_model.pt",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
)

trainer.fit(callbacks=[checkpoint])

# 取得最佳模型路徑
print(f"Best model: {checkpoint.best_path}")
```

### ProgressLogger - 進度日誌

記錄訓練進度。

```python
from ivit.train import ProgressLogger

logger = ProgressLogger(log_frequency=10)  # 每 10 個 batch 記錄一次

trainer.fit(callbacks=[logger])
```

### LRScheduler - 學習率調度

動態調整學習率。

```python
from ivit.train import LRScheduler

# Step decay
scheduler = LRScheduler("step", step_size=10, gamma=0.1)

# Cosine annealing
scheduler = LRScheduler("cosine", T_max=100)

# Reduce on plateau
scheduler = LRScheduler("plateau", patience=5, factor=0.5)

trainer.fit(callbacks=[scheduler])
```

### TensorBoardLogger - TensorBoard 記錄

將訓練指標記錄到 TensorBoard。

```python
from ivit.train import TensorBoardLogger

tb_logger = TensorBoardLogger(log_dir="runs/experiment1")

trainer.fit(callbacks=[tb_logger])

# 啟動 TensorBoard
# tensorboard --logdir=runs/
```

### 組合使用

```python
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10),
    ModelCheckpoint("checkpoints/best.pt", monitor="val_loss"),
    LRScheduler("cosine", T_max=50),
    TensorBoardLogger("runs/experiment1"),
]

trainer.fit(callbacks=callbacks)
```

---

## 模型匯出

訓練完成後，將模型匯出為部署格式。

### 匯出 ONNX（推薦）

```python
# 基本匯出
trainer.export("model.onnx")

# 指定輸入尺寸
trainer.export(
    "model.onnx",
    format="onnx",
    input_shape=(1, 3, 224, 224),  # (batch, channels, height, width)
)

# FP16 量化（減小模型大小）
trainer.export(
    "model_fp16.onnx",
    format="onnx",
    quantize="fp16",
)
```

### 匯出 TorchScript

```python
trainer.export("model.pt", format="torchscript")
```

### 匯出 OpenVINO IR

```python
# FP32
trainer.export("model.xml", format="openvino")

# FP16
trainer.export("model_fp16.xml", format="openvino", quantize="fp16")
```

### 匯出 TensorRT

```python
# FP16 優化
trainer.export("model.engine", format="tensorrt", quantize="fp16")
```

---

## 進階設定

### 自訂資料增強

```python
from ivit.train import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    ColorJitter,
    Normalize,
    ToTensor,
)

train_transform = Compose([
    Resize(224),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2),
    Normalize(),
    ToTensor(),
])

# 套用到資料集
dataset = ImageFolderDataset(
    "./my_dataset",
    transforms=train_transform,
)
```

### 使用自訂模型

```python
import torch.nn as nn
import torchvision.models as models

# 載入預訓練模型並修改
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights='DEFAULT')
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.backbone(x)

model = CustomModel(num_classes=10)

trainer = Trainer(
    model=model,  # 傳入自訂模型
    dataset=dataset,
    epochs=20,
    device="cuda:0",
)
```

### 恢復訓練

```python
import torch

# 載入 checkpoint
checkpoint = torch.load("checkpoints/best_model.pt")

# 恢復模型
trainer.model.load_state_dict(checkpoint['model_state_dict'])
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 繼續訓練
trainer.fit()
```

---

## 常見問題

### Q: PyTorch CUDA 版本與系統 CUDA 版本衝突？

**症狀**：出現 `nvjitlink` 相關錯誤或 CUDA 初始化失敗。

**解決方案**：

1. **安裝對應版本 PyTorch（推薦）**：

   ```bash
   # 查詢系統 CUDA 版本
   nvcc --version

   # 安裝對應版本 PyTorch
   # CUDA 12.6
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

   # CUDA 12.4
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

   # CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **使用 CPU 版本（CI 環境）**：

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **設定 LD_PRELOAD（臨時解決）**：

   ```bash
   # 找到 nvjitlink 庫位置
   find /usr -name "libnvJitLink.so*" 2>/dev/null

   # 設定環境變數
   export LD_PRELOAD=/path/to/libnvJitLink.so.12
   ```

### Q: 訓練速度太慢？

**解決方案**：

1. **增加 num_workers**：

   ```python
   trainer = Trainer(..., num_workers=8)
   ```

2. **使用較小的模型**：

   ```python
   trainer = Trainer(model="mobilenet_v3_small", ...)
   ```

3. **凍結骨幹網路**：

   ```python
   trainer = Trainer(..., freeze_backbone=True)
   ```

4. **使用混合精度訓練**：

   將在未來版本支援。

### Q: 記憶體不足 (OOM)？

**解決方案**：

1. **減小 batch_size**：

   ```python
   trainer = Trainer(..., batch_size=16)  # 或更小
   ```

2. **使用較小的輸入尺寸**：

   修改資料增強中的 Resize 大小。

3. **使用梯度累積**：

   將在未來版本支援。

### Q: 如何處理不平衡資料集？

**解決方案**：

1. **使用加權採樣**：

   將在未來版本支援 WeightedRandomSampler。

2. **資料增強**：

   對少數類別進行更多增強。

3. **調整損失函式權重**：

   將在未來版本支援。

---

## 下一步

- [API 規格](../api/api-spec.md) - 完整 API 文件
- [模型格式指南](../getting-started.md#模型格式) - 模型轉換與優化
- [部署指南](../deployment/) - 模型部署

---

*iVIT-SDK Training Module - Innodisk Corporation*
