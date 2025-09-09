# iVIT 2.0 SDK - 完整使用指南

## 📋 目錄

1. [安裝指南](#安裝指南)
2. [快速開始](#快速開始)
3. [訓練使用指南](#訓練使用指南)
4. [進階配置](#進階配置)
5. [故障排除](#故障排除)
6. [性能優化](#性能優化)
7. [API參考](#api參考)

---

## 安裝指南

### 系統需求

#### 基本需求
- **Python**: 3.8+ (推薦 3.9 或 3.10)
- **操作系統**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **記憶體**: 最少 8GB RAM (推薦 16GB+)
- **硬碟空間**: 5GB 可用空間

#### GPU需求 (推薦)
- **NVIDIA GPU**: CUDA 11.8+ 支援
- **GPU記憶體**: 最少 4GB VRAM (推薦 8GB+)
- **驅動程式**: 最新的NVIDIA驅動程式

### 安裝步驟

#### 方法 1: 從源碼安裝 (推薦)

```bash
# 1. 克隆專案
git clone <repository-url>
cd ivit_2.0_sdk

# 2. 建立虛擬環境 (推薦)
python -m venv ivit_env

# 3. 啟動虛擬環境
# Windows:
ivit_env\Scripts\activate
# Linux/macOS:
source ivit_env/bin/activate

# 4. 升級pip
pip install --upgrade pip

# 5. 安裝依賴
pip install -r requirements.txt

# 6. 安裝SDK (開發模式)
pip install -e .
```

#### 方法 2: 使用pip安裝 (未來支援)

```bash
# 將來會支援直接從PyPI安裝
pip install ivit-sdk
```

### 環境配置

#### CUDA設置 (GPU用戶)

1. **檢查CUDA版本**
```bash
nvidia-smi
nvcc --version
```

2. **安裝對應的PyTorch版本**
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU版本 (不推薦用於訓練)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 驗證安裝

```python
# 測試基本安裝
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU數量: {torch.cuda.device_count()}")

# 測試iVIT SDK
try:
    from ivit.trainer.classification import ClassificationConfig
    from ivit.trainer.detection import DetectionConfig  
    from ivit.trainer.segmentation import SegmentationConfig
    from ivit.utils.smart_recommendation import SmartRecommendationEngine
    print("✅ iVIT 2.0 SDK 安裝成功！")
except ImportError as e:
    print(f"❌ 安裝失敗: {e}")
```

---

## 快速開始

### 創建測試資料

```bash
cd ivit_2.0_sdk
python examples/create_test_data.py
```

### 運行簡單範例

```bash
# 修改examples/beginner/classification_simple.py中的資料路徑
# 然後執行：
python examples/beginner/classification_simple.py
```

---

## 訓練使用指南

### 支援的訓練類型

| 類型 | 腳本 | 資料格式 | 支援模型 | 單卡 | 多卡 |
|------|------|----------|----------|------|------|
| **Classification** | `classification_training.py` | ImageFolder | ResNet, EfficientNet, DenseNet | ✅ | ✅ |
| **Detection** | `detection_training.py` | YOLO | YOLOv8 系列 | ✅ | ✅ |
| **Segmentation** | `segmentation_training.py` | YOLO | YOLOv8-seg 系列 | ✅ | ✅ |
| **Custom** | `custom_training.py` | ImageFolder | 自定義模型 | ✅ | ✅ |

### 統一訓練腳本 (推薦)

使用統一的 `run_training.py` 腳本，支援所有訓練類型：

```bash
# 基本用法
python examples/beginner/run_training.py --task <任務類型> --data_path <資料集路徑>

# 完整參數
python examples/beginner/run_training.py \
    --task classification \
    --data_path /path/to/dataset \
    --device 0,1 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.01
```

### 個別訓練腳本

也可以直接使用各類型的專用腳本：

```bash
# 分類訓練
python examples/beginner/classification_training.py --data_path /path/to/dataset --device 0,1

# 偵測訓練
python examples/beginner/detection_training.py --data_path /path/to/dataset --device 0,1

# 分割訓練
python examples/beginner/segmentation_training.py --data_path /path/to/dataset --device 0,1

# 自定義訓練
python examples/beginner/custom_training.py --data_path /path/to/dataset --device 0,1
```

### 設備配置

#### 單卡訓練
```bash
--device 0
```

#### 多卡訓練
```bash
# 使用 GPU 0 和 1
--device 0,1

# 使用 GPU 0, 1, 2, 3
--device 0,1,2,3
```

### 資料集格式

#### Classification & Custom (ImageFolder)
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

#### Detection & Segmentation (YOLO)
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

### 完整使用範例

```bash
# 1. 分類訓練 (ResNet50, 多卡)
python examples/beginner/run_training.py \
    --task classification \
    --data_path /path/to/classification/dataset \
    --device 0,1,2,3 \
    --epochs 100 \
    --batch_size 32 \
    --model resnet50 \
    --num_classes 10

# 2. 偵測訓練 (YOLOv8n, 單卡)
python examples/beginner/run_training.py \
    --task detection \
    --data_path /path/to/yolo/dataset \
    --device 0 \
    --epochs 50 \
    --model yolov8n.pt \
    --img_size 640

# 3. 分割訓練 (YOLOv8n-seg, 多卡)
python examples/beginner/run_training.py \
    --task segmentation \
    --data_path /path/to/yolo/dataset \
    --device 0,1 \
    --epochs 50 \
    --model yolov8n-seg.pt

# 4. 自定義訓練
python examples/beginner/run_training.py \
    --task custom \
    --data_path /path/to/dataset \
    --device 0 \
    --epochs 50 \
    --num_classes 5 \
    --img_size 224
```

### 多卡訓練詳解

#### DataParallel (DP) 模式
- 自動啟用當 `device` 包含多個 GPU 時
- 適合單機多卡訓練
- 使用 `nn.DataParallel` 實現

#### DistributedDataParallel (DDP) 模式
- 使用 `torchrun` 啟動
- 適合多機多卡訓練
- 更好的擴展性和穩定性

**DDP 使用範例**:
```bash
# 使用 torchrun 啟動 DDP 訓練
torchrun --nproc_per_node=4 examples/beginner/run_training.py \
    --task classification \
    --data_path /path/to/dataset \
    --device 0,1,2,3 \
    --epochs 100
```

### 訓練參數說明

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

### 訓練監控

#### 即時監控
```bash
# 監控 GPU 使用率
nvidia-smi -l 1

# 監控訓練進度
tail -f training.log
```

#### 訓練日誌
- 訓練過程會顯示詳細的進度信息
- 包含損失值、準確率、學習率等指標
- 支援 TensorBoard 可視化 (如果配置)

---

## 進階配置

### 自定義損失函數

```python
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 自定義損失計算
        pass

# 在訓練器中使用
trainer = ExpertClassificationTrainer(config, custom_loss=CustomLoss())
```

### 自定義學習率調度

```python
from torch.optim.lr_scheduler import StepLR

def custom_scheduler(optimizer):
    return StepLR(optimizer, step_size=30, gamma=0.1)

trainer = ExpertClassificationTrainer(config, custom_scheduler=custom_scheduler)
```

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

### 智能推薦系統

```python
from ivit.utils.dataset_analyzer import DatasetAnalyzer
from ivit.utils.smart_recommendation import SmartRecommendationEngine

# 分析資料集
analyzer = DatasetAnalyzer()
stats = analyzer.analyze_classification_dataset("path/to/data")

# 獲取推薦
recommender = SmartRecommendationEngine()
recommendations = recommender.get_recommendations('classification', stats)

print(f"推薦模型: {recommendations['model']}")
print(f"推薦學習率: {recommendations['learning_rate']}")
```

---

## 故障排除

### 常見問題

#### 1. PyTorch安裝問題

**問題**: `pip install torch` 很慢或失敗
**解決**:
```bash
# 使用國內鏡像
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或指定PyTorch官方源
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. CUDA版本不匹配

**問題**: `RuntimeError: CUDA runtime error`
**解決**:
1. 檢查CUDA版本: `nvidia-smi`
2. 安裝對應版本的PyTorch
3. 重新啟動Python環境

#### 3. 記憶體不足

**問題**: `CUDA out of memory`
**解決**:
```python
# 減小批次大小
config.batch_size = 16  # 或更小

# 使用混合精度訓練
config.mixed_precision = True
```

#### 4. 模組找不到

**問題**: `ModuleNotFoundError: No module named 'ivit'`
**解決**:
```bash
# 確保在正確的虛擬環境中
which python
pip list | grep ivit

# 重新安裝SDK
pip uninstall ivit-sdk
pip install -e .
```

#### 5. YOLOv8相關問題

**問題**: ultralytics安裝或使用問題
**解決**:
```bash
# 更新ultralytics
pip install --upgrade ultralytics

# 清除快取
pip cache purge
```

#### 6. 多卡訓練問題

**問題**: 多卡訓練不工作
**解決**:
- 確保所有 GPU 可見
- 檢查 CUDA 環境配置
- 使用 `CUDA_VISIBLE_DEVICES` 環境變數

#### 7. 資料集格式錯誤

**問題**: 資料集格式不正確
**解決**:
- 檢查資料集結構
- 確認標註檔案格式
- 驗證路徑配置

### 調試模式

```bash
# 啟用詳細日誌
export PYTHONPATH=/path/to/ivit:$PYTHONPATH
python examples/beginner/run_training.py --task classification --data_path /path/to/dataset --device 0 --epochs 1
```

### 調試工具
- 測試腳本: `test_all_training.py`
- 詳細日誌: 啟用 debug 模式
- 監控工具: `nvidia-smi`, `tail -f`

---

## 性能優化

### GPU記憶體優化

```python
# 配置文件中的優化設定
config.batch_size = 32        # 根據GPU記憶體調整
config.num_workers = 4        # 資料載入線程數
config.pin_memory = True      # 加速GPU傳輸
config.mixed_precision = True # 混合精度訓練
```

### CPU優化

```python
# CPU訓練優化
import torch
torch.set_num_threads(4)  # 設定CPU線程數

config.batch_size = 16    # CPU訓練建議較小批次
config.num_workers = 2    # 減少資料載入線程
```

### 訓練加速

1. **多 GPU 並行訓練**
2. **資料載入優化**
3. **快取機制**
4. **混合精度訓練**

### 監控和調試

- 詳細的訓練日誌
- GPU 使用率監控
- 進度條顯示

---

## API參考

### 核心類別

#### BaseTrainer
```python
class BaseTrainer:
    def __init__(self, config: TaskConfig)
    def create_model(self) -> nn.Module
    def get_dataloader(self, dataset_path: str, batch_size: int, split: str = 'train') -> DataLoader
    def train(self, dataset_path: str, epochs: int, batch_size: int) -> Dict[str, Any]
```

#### TaskConfig
```python
class TaskConfig:
    def __init__(self, model_name: str, img_size: int, learning_rate: float, device: str)
```

### 訓練器類別

#### ClassificationTrainer
```python
class ClassificationTrainer(BaseTrainer):
    def __init__(self, model_name: str, img_size: int, num_classes: int, 
                 learning_rate: float, device: str)
```

#### DetectionTrainer
```python
class DetectionTrainer(BaseTrainer):
    def __init__(self, model_name: str, img_size: int, learning_rate: float, device: str)
    def validate_dataset_format(self, dataset_path: str) -> bool
```

#### SegmentationTrainer
```python
class SegmentationTrainer(BaseTrainer):
    def __init__(self, model_name: str, img_size: int, learning_rate: float, device: str)
```

### 工具類別

#### DatasetAnalyzer
```python
class DatasetAnalyzer:
    def analyze_classification_dataset(self, dataset_path: str) -> Dict[str, Any]
    def analyze_detection_dataset(self, dataset_path: str) -> Dict[str, Any]
    def extract_detection_statistics(self, dataset_path: str) -> Dict[str, Any]
```

#### SmartRecommendationEngine
```python
class SmartRecommendationEngine:
    def get_recommendations(self, task_type: str, stats: Dict[str, Any]) -> Dict[str, Any]
    def get_classification_recommendations(self, stats: Dict[str, Any]) -> Dict[str, Any]
    def get_detection_recommendations(self, stats: Dict[str, Any]) -> Dict[str, Any]
```

---

## 升級指南

### 升級到新版本
```bash
# 拉取最新代碼
git pull origin main

# 更新依賴
pip install -r requirements.txt --upgrade

# 重新安裝SDK
pip install -e . --force-reinstall
```

### 檢查版本
```python
import ivit
print(ivit.__version__)
```

---

## 不同平台安裝

### Windows 10/11

1. **安裝Python 3.9+**
   - 從 python.org 下載並安裝
   - 確保勾選 "Add Python to PATH"

2. **安裝Git**
   - 從 git-scm.com 下載並安裝

3. **安裝CUDA (GPU用戶)**
   - 從NVIDIA官網下載CUDA Toolkit
   - 安裝對應版本的cuDNN

4. **按照上述步驟安裝SDK**

### Ubuntu 18.04+

```bash
# 更新套件管理器
sudo apt update && sudo apt upgrade -y

# 安裝Python和pip
sudo apt install python3.9 python3.9-pip python3.9-venv -y

# 安裝Git
sudo apt install git -y

# 安裝CUDA (GPU用戶)
# 參考NVIDIA官方文檔

# 按照上述步驟安裝SDK
```

### macOS 10.15+

```bash
# 安裝Homebrew (如果沒有)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安裝Python
brew install python@3.9

# 安裝Git
brew install git

# 注意：macOS不支援CUDA，只能使用CPU版本
# 按照上述步驟安裝SDK (CPU版本)
```

---

## 獲取幫助

如果安裝過程中遇到問題：

1. **檢查系統日誌**: 查看詳細錯誤訊息
2. **搜索已知問題**: 查看GitHub Issues
3. **提交新問題**: 包含系統信息和錯誤日誌
4. **社群討論**: 參與GitHub Discussions

### 提交問題時請提供：
- 操作系統和版本
- Python版本
- CUDA版本 (如果使用GPU)
- 完整的錯誤訊息
- 安裝步驟

---

**祝您使用愉快！開始您的AI模型訓練之旅吧！** 🚀
