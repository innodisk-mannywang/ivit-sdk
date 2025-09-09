# iVIT 2.0 SDK

🚀 **新一代AI視覺訓練與部署SDK** - 讓AI模型訓練變得簡單而強大！

## ✨ 核心特色

### 🎯 多任務支援
- **🖼️ 圖像分類**: 支援ResNet、EfficientNet、MobileNet、Vision Transformer等主流模型
- **🎯 物件偵測**: 基於YOLOv8的最新物件偵測技術
- **🖌️ 語義分割**: DeepLabV3、FCN、自定義U-Net等分割模型

### 🧠 智慧推薦系統
- **📊 自動資料集分析**: 智能分析資料集特徵和複雜度
- **🎚️ 參數自動推薦**: 根據資料集特性推薦最佳超參數
- **📈 性能優化建議**: 提供專業的模型選擇和訓練建議

### 👥 多層級API設計
- **🌟 初學者模式**: 一鍵訓練，預設最佳配置
- **⚡ 進階模式**: 靈活配置，智能推薦輔助
- **🎓 專家模式**: 完全自定義，支援複雜訓練流程

### 🌐 跨平台部署
- **NVIDIA**: TensorRT優化
- **Intel**: OpenVINO加速  
- **Xilinx**: Vitis AI支援
- **Qualcomm**: SNPE整合

## 🚀 快速開始

### 安裝

```bash
# 克隆專案
git clone <repository-url>
cd ivit_2.0_sdk

# 建立虛擬環境 (推薦)
python3 -m venv ivit_env
source ivit_env/bin/activate

# 安裝SDK
pip install -e .
```

#### 程式碼方式

```python
from ivit.trainer.classification import ClassificationTrainer

# 創建訓練器
trainer = ClassificationTrainer(
    model_name='resnet18',
    img_size=224,
    num_classes=10,
    learning_rate=0.01,
    device='0,1'
)

# 開始訓練
results = trainer.train(
    dataset_path='/path/to/dataset',
    epochs=50,
    batch_size=16
)

print("🎉 訓練完成！")
```

## 📚 使用範例

### 🌟 初學者範例

**💡 推薦使用統一訓練腳本（自動檢測參數）：**

```bash
# 統一訓練腳本 - 自動檢測類別數量和圖片尺寸
python examples/beginner/run_training.py --task classification --data_path /path/to/dataset --device 0,1 --epochs 50
python examples/beginner/run_training.py --task detection --data_path /path/to/yolo/dataset --device 0,1 --epochs 50
python examples/beginner/run_training.py --task segmentation --data_path /path/to/yolo/dataset --device 0,1 --epochs 50
```

**手動指定參數（進階用戶）：**

```bash
# 分類訓練 (需要指定類別數量)
python examples/beginner/classification_training.py --data_path /path/to/dataset --device 0,1 --epochs 50 --num_classes 10

# 偵測訓練 (類別數量從 data.yaml 自動讀取)
python examples/beginner/detection_training.py --data_path /path/to/yolo/dataset --device 0,1 --epochs 50

# 分割訓練 (類別數量從 data.yaml 自動讀取)
python examples/beginner/segmentation_training.py --data_path /path/to/yolo/dataset --device 0,1 --epochs 50

# 自定義訓練 (需要指定類別數量)
python examples/beginner/custom_training.py --data_path /path/to/dataset --device 0,1 --epochs 50 --num_classes 10
```

### ⚡ 進階範例 - 智能推薦系統

#### 🧠 智能推薦系統功能

智能推薦系統會自動分析您的資料集並推薦最佳訓練參數，包括：
- **模型選擇**: 根據資料集大小和複雜度推薦最適合的模型
- **超參數調優**: 自動推薦學習率、批次大小、訓練輪數等
- **資料增強策略**: 提供針對性的資料增強建議
- **性能優化**: 基於資料集特性提供專業建議

#### 🚀 快速開始

**1. 創建測試資料集**
```bash
python create_test_data.py
```

**2. 執行智能推薦範例**

**方式一：互動模式（推薦新手）**
```bash
cd examples/advanced
python3 smart_recommendation_example.py
```

**方式二：命令行參數模式（推薦進階用戶）**
```bash
# 分類任務
python3 smart_recommendation_example.py --task classification --data_path /path/to/dataset

# 偵測任務
python3 smart_recommendation_example.py --task detection --data_path /path/to/data.yaml

# 查看幫助
python3 smart_recommendation_example.py --help
```

#### 💻 程式碼使用方式

```python
from ivit.utils.dataset_analyzer import DatasetAnalyzer
from ivit.utils.smart_recommendation import SmartRecommendationEngine

# 分析資料集
analyzer = DatasetAnalyzer()
stats = analyzer.analyze_classification_dataset("path/to/data")

# 獲取智能推薦
recommender = SmartRecommendationEngine() 
recommendations = recommender.get_recommendations('classification', stats)

print(f"推薦模型: {recommendations['model']}")
print(f"推薦學習率: {recommendations['learning_rate']}")
print(f"推薦批次大小: {recommendations['batch_size']}")
print(f"推薦訓練輪數: {recommendations['epochs']}")
```

#### 📊 支援的任務類型

| 任務類型 | 資料格式 | 支援模型 | 自動分析 |
|---------|----------|----------|----------|
| **分類** | ImageFolder | ResNet, EfficientNet, MobileNet, ViT | ✅ |
| **偵測** | YOLO | YOLOv8 系列 | ✅ |
| **分割** | YOLO | YOLOv8-seg 系列 | ✅ |

#### 🎯 推薦結果說明

智能推薦系統會提供：
- **模型架構**: 根據資料集大小和複雜度選擇
- **學習率**: 基於資料集特性優化
- **批次大小**: 考慮GPU記憶體和圖片解析度
- **訓練輪數**: 根據資料集大小調整
- **資料增強**: 針對性的增強策略
- **推薦原因**: 詳細解釋每個推薦的依據

## 🎨 測試資料生成

快速生成測試資料集：

```bash
python examples/create_test_data.py
```

這將創建：
- 分類測試資料集 (5類，每類20張圖片)
- YOLO格式偵測資料集 (3類，50張圖片) 
- 分割測試資料集 (4類，30張圖片)

## 🏗️ 架構設計

```
ivit_2.0_sdk/
├── ivit/                          # 核心SDK
│   ├── core/                      # 基礎框架
│   │   ├── base_trainer.py        # 通用訓練器基類
│   │   └── task_config.py         # 任務配置基類
│   ├── trainer/                   # 訓練器實作
│   │   ├── classification.py      # 分類訓練器
│   │   ├── detection.py           # 偵測訓練器  
│   │   └── segmentation.py        # 分割訓練器
│   ├── utils/                     # 輔助工具
│   │   ├── dataset_analyzer.py    # 資料集分析器
│   │   └── smart_recommendation.py # 智能推薦引擎
│   └── deployment/                # 部署模組 (規劃中)
├── examples/                      # 使用範例
│   ├── beginner/                  # 初學者範例
│   ├── advanced/                  # 進階範例  
│   ├── expert/                    # 專家範例
│   └── create_test_data.py        # 測試資料生成器
└── docs/                          # 文檔
```

## 📋 支援的模型

### 分類模型
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0-B7)
- MobileNet V2/V3
- Vision Transformer (ViT)
- DenseNet

### 偵測模型  
- YOLOv8 (n, s, m, l, x)
- 支援COCO格式和自定義資料集

### 分割模型
- DeepLabV3 (ResNet backbone)
- FCN (ResNet backbone) 
- U-Net (自定義實作)

## 🚀 訓練支援

### 支援的訓練類型

| 類型 | 資料格式 | 支援模型 | 單卡 | 多卡 |
|------|----------|----------|------|------|
| **Classification** | ImageFolder | ResNet, EfficientNet, DenseNet | ✅ | ✅ |
| **Detection** | YOLO | YOLOv8 系列 | ✅ | ✅ |
| **Segmentation** | YOLO | YOLOv8-seg 系列 | ✅ | ✅ |
| **Custom** | ImageFolder | 自定義模型 | ✅ | ✅ |

### 多卡訓練

```bash
# DataParallel (自動)
python run_training.py --task classification --data_path /path/to/dataset --device 0,1,2,3

# DistributedDataParallel (手動)
torchrun --nproc_per_node=4 run_training.py --task classification --data_path /path/to/dataset --device 0,1,2,3
```

## 🔍 常見問題

**Q: 如何選擇合適的模型？**
A: 使用智能推薦系統！它會根據你的資料集自動推薦最適合的模型。

**Q: 訓練很慢怎麼辦？**
A: 1) 減少批次大小 2) 使用較小的模型 3) 減少訓練輪數 4) 檢查GPU使用率

**Q: 支援哪些資料格式？**
A: 
- 分類: ImageFolder格式 (train/class1/, train/class2/)
- 偵測: YOLO格式 (images/, labels/, data.yaml)
- 分割: images/, masks/ 格式

## 📚 詳細文檔

- **[完整使用指南](docs/COMPLETE_GUIDE.md)** - 詳細的安裝、配置和使用說明
- **[訓練使用指南](examples/beginner/README_TRAINING.md)** - 詳細的訓練使用說明和故障排除
- **[API文檔](docs/api.md)** - 完整的API參考
- **[範例集合](examples/)** - 豐富的使用範例

## 📝 更新日誌

### v2.0.0 (2024-01-01)
- ✨ 首次發布
- 🎯 支援分類、偵測、分割三大任務
- 🧠 智能推薦系統
- 👥 多層級API設計
- 🎨 完整使用範例
- ✅ 多GPU訓練支援

## 🤝 貢獻指南

歡迎提交Issue和Pull Request！

### 開發設置
```bash
git clone <repository-url>
cd ivit_2.0_sdk
pip install -r requirements.txt
pip install -e .
```

### 運行測試
```bash
python examples/create_test_data.py
python examples/beginner/test_all_training.py --data_path /path/to/test/dataset
```

## 📄 授權

MIT License

## 🆘 技術支援

- **文檔**: [完整使用指南](docs/COMPLETE_GUIDE.md)
- **問題回報**: [GitHub Issues]  
- **討論區**: [GitHub Discussions]

---

**讓AI模型訓練變得更簡單、更智能、更高效！** 🚀

---

*iVIT 2.0 SDK - Powered by PyTorch & YOLOv8*