# iVIT 2.0 SDK - 訓練使用指南

本指南介紹如何使用 iVIT 2.0 SDK 進行各種類型的機器學習訓練，支援單卡和多卡訓練。

## 📋 支援的訓練類型

- **Classification** - 圖像分類
- **Detection** - 物件偵測
- **Segmentation** - 語義分割
- **Custom** - 自定義模型訓練

## 🚀 快速開始

### 1. 統一訓練腳本 (推薦)

使用統一的 `run_training.py` 腳本，支援所有訓練類型：

```bash
# 基本用法
python run_training.py --task <任務類型> --data_path <資料集路徑>

# 完整參數
python run_training.py \
    --task classification \
    --data_path /path/to/dataset \
    --device 0,1 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.01
```

### 2. 個別訓練腳本

也可以直接使用各類型的專用腳本：

```bash
# 分類訓練
python classification_training.py --data_path /path/to/dataset --device 0,1

# 偵測訓練
python detection_training.py --data_path /path/to/dataset --device 0,1

# 分割訓練
python segmentation_training.py --data_path /path/to/dataset --device 0,1

# 自定義訓練
python custom_training.py --data_path /path/to/dataset --device 0,1
```

## 🔧 設備配置

### 單卡訓練
```bash
--device 0
```

### 多卡訓練
```bash
# 使用 GPU 0 和 1
--device 0,1

# 使用 GPU 0, 1, 2, 3
--device 0,1,2,3
```

## 📊 各類型訓練詳細說明

### 1. Classification (圖像分類)

**資料集格式**: ImageFolder 格式
```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image1.jpg
│       └── image2.jpg
└── val/
    ├── class1/
    └── class2/
```

**使用範例**:
```bash
# 單卡訓練
python run_training.py \
    --task classification \
    --data_path /path/to/classification/dataset \
    --device 0 \
    --epochs 100 \
    --model resnet50 \
    --num_classes 10

# 多卡訓練
python run_training.py \
    --task classification \
    --data_path /path/to/classification/dataset \
    --device 0,1,2,3 \
    --epochs 100 \
    --batch_size 32
```

**支援的模型**:
- `resnet18`, `resnet50`, `resnet101`
- `efficientnet_b0`, `efficientnet_b1`
- `densenet121`, `densenet161`

### 2. Detection (物件偵測)

**資料集格式**: YOLO 格式
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image1.jpg
│       └── image2.jpg
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   └── image2.txt
│   └── val/
│       ├── image1.txt
│       └── image2.txt
└── data.yaml
```

**data.yaml 範例**:
```yaml
train: ../images/train
val: ../images/val
nc: 3
names: ['class1', 'class2', 'class3']
```

**使用範例**:
```bash
# 單卡訓練
python run_training.py \
    --task detection \
    --data_path /path/to/yolo/dataset \
    --device 0 \
    --epochs 100 \
    --model yolov8n.pt

# 多卡訓練
python run_training.py \
    --task detection \
    --data_path /path/to/yolo/dataset \
    --device 0,1,2,3 \
    --epochs 100 \
    --batch_size 32
```

**支援的模型**:
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`

### 3. Segmentation (語義分割)

**資料集格式**: YOLO 格式 (與 Detection 相同)

**使用範例**:
```bash
# 單卡訓練
python run_training.py \
    --task segmentation \
    --data_path /path/to/yolo/dataset \
    --device 0 \
    --epochs 100 \
    --model yolov8n-seg.pt

# 多卡訓練
python run_training.py \
    --task segmentation \
    --data_path /path/to/yolo/dataset \
    --device 0,1,2,3 \
    --epochs 100
```

**支援的模型**:
- `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`

### 4. Custom (自定義模型)

**資料集格式**: ImageFolder 格式 (與 Classification 相同)

**使用範例**:
```bash
# 單卡訓練
python run_training.py \
    --task custom \
    --data_path /path/to/dataset \
    --device 0 \
    --epochs 100 \
    --num_classes 5 \
    --img_size 224

# 多卡訓練
python run_training.py \
    --task custom \
    --data_path /path/to/dataset \
    --device 0,1,2,3 \
    --epochs 100 \
    --batch_size 32
```

## 📈 訓練參數說明

| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--task` | 訓練任務類型 | 必填 | `classification`, `detection`, `segmentation`, `custom` |
| `--data_path` | 資料集路徑 | 必填 | `/path/to/dataset` |
| `--device` | 設備配置 | `0` | `0` (單卡), `0,1` (多卡) |
| `--epochs` | 訓練輪數 | `50` | `100` |
| `--batch_size` | 批次大小 | `16` | `32` |
| `--learning_rate` | 學習率 | `0.01` | `0.001` |
| `--model` | 模型名稱 | 依任務而定 | `resnet50`, `yolov8n.pt` |
| `--img_size` | 圖片尺寸 | 依任務而定 | `224`, `640` |
| `--num_classes` | 類別數量 | 依任務而定 | `10` |

## 🔄 多卡訓練說明

### DataParallel (DP) 模式
- 自動啟用當 `device` 包含多個 GPU 時
- 適合單機多卡訓練
- 使用 `nn.DataParallel` 實現

### DistributedDataParallel (DDP) 模式
- 使用 `torchrun` 啟動
- 適合多機多卡訓練
- 更好的擴展性和穩定性

**DDP 使用範例**:
```bash
# 使用 torchrun 啟動 DDP 訓練
torchrun --nproc_per_node=4 run_training.py \
    --task classification \
    --data_path /path/to/dataset \
    --device 0,1,2,3 \
    --epochs 100
```

## 📊 訓練監控

### 即時監控
```bash
# 監控 GPU 使用率
nvidia-smi -l 1

# 監控訓練進度
tail -f training.log
```

### 訓練日誌
- 訓練過程會顯示詳細的進度信息
- 包含損失值、準確率、學習率等指標
- 支援 TensorBoard 可視化 (如果配置)

## 🛠️ 故障排除

### 常見問題

1. **CUDA 記憶體不足**
   - 減少 `batch_size`
   - 減少 `img_size`
   - 使用梯度累積

2. **多卡訓練問題**
   - 確保所有 GPU 可見
   - 檢查 CUDA 環境配置
   - 使用 `CUDA_VISIBLE_DEVICES` 環境變數

3. **資料集格式錯誤**
   - 檢查資料集結構
   - 確認標註檔案格式
   - 驗證路徑配置

### 調試模式
```bash
# 啟用詳細日誌
export PYTHONPATH=/path/to/ivit:$PYTHONPATH
python run_training.py --task classification --data_path /path/to/dataset --device 0 --epochs 1
```

## 📚 進階使用

### 自定義訓練器
參考 `custom_training.py` 範例，創建自己的訓練器：

```python
from ivit.core.base_trainer import BaseTrainer

class MyCustomTrainer(BaseTrainer):
    def create_model(self):
        # 實現自定義模型
        pass
    
    def get_dataloader(self, dataset_path, batch_size, split='train'):
        # 實現自定義資料載入
        pass
```

### 模型保存和載入
訓練完成後，模型會自動保存到指定目錄，可以後續用於推理。

## 🤝 支援

如有問題，請參考：
- 專案文檔
- 範例代碼
- 問題回報

---

**注意**: 請確保已正確安裝所有依賴項，並配置好 CUDA 環境。
