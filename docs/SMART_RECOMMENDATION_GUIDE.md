# 智能推薦系統使用指南

## 🚀 快速開始

### 1. 創建測試資料集

首先，創建測試資料集以便立即執行範例：

```bash
cd /home/ipa-genai/Workspace/Project/ivit-sdk
python create_test_data.py
```

這將創建：
- 分類測試資料集：`test_data/classification/` (5類，每類20張訓練圖片，5張驗證圖片)
- 偵測測試資料集：`test_data/detection/` (3類，30張訓練圖片，10張驗證圖片)

### 2. 執行智能推薦範例

#### 方式一：互動模式（推薦新手使用）

```bash
cd examples/advanced
python3 smart_recommendation_example.py
```

選擇任務類型：
- 輸入 `1` 進行分類任務
- 輸入 `2` 進行物件偵測任務

然後選擇資料集來源：
- 輸入 `1` 使用預設測試資料集
- 輸入 `2` 輸入自定義資料集路徑

#### 方式二：命令行參數（推薦進階用戶）

```bash
# 分類任務
python3 smart_recommendation_example.py --task classification --data_path /path/to/your/dataset

# 偵測任務  
python3 smart_recommendation_example.py --task detection --data_path /path/to/your/data.yaml

# 查看幫助
python3 smart_recommendation_example.py --help
```

#### 方式三：強制互動模式

```bash
python3 smart_recommendation_example.py --interactive
```

## 📊 功能說明

### 智能推薦系統會自動：

1. **分析資料集特徵**：
   - 類別數量
   - 圖片總數
   - 平均圖片大小
   - 資料集複雜度評分

2. **推薦最佳參數**：
   - 模型架構 (ResNet, EfficientNet, MobileNet, YOLO等)
   - 學習率
   - 批次大小
   - 訓練輪數
   - 資料增強策略
   - 正則化技術

3. **提供推薦原因**：
   - 解釋為什麼選擇這些參數
   - 基於資料集特性的分析

## 🎯 支援的任務類型

### 分類任務
- 支援模型：ResNet, EfficientNet, MobileNet, Vision Transformer
- 資料格式：ImageFolder結構 (train/class1/, train/class2/)
- 自動檢測類別數量和圖片尺寸

### 物件偵測任務
- 支援模型：YOLOv8系列 (n, s, m, l, x)
- 資料格式：YOLO格式 (images/, labels/, data.yaml)
- 自動分析物件數量和複雜度

## 🔧 自定義使用

### 使用自己的資料集

#### 方法一：使用命令行參數（推薦）

```bash
# 使用自己的分類資料集
python3 smart_recommendation_example.py --task classification --data_path /path/to/your/classification/dataset

# 使用自己的偵測資料集
python3 smart_recommendation_example.py --task detection --data_path /path/to/your/data.yaml
```

#### 方法二：使用 Python API

1. **分類資料集**：
   ```python
   from ivit.utils.dataset_analyzer import DatasetAnalyzer
   from ivit.utils.smart_recommendation import SmartRecommendationEngine
   
   # 分析資料集
   analyzer = DatasetAnalyzer()
   stats = analyzer.analyze_classification_dataset("path/to/your/dataset/train")
   
   # 獲取推薦
   recommender = SmartRecommendationEngine()
   recommendations = recommender.get_recommendations('classification', stats)
   ```

2. **偵測資料集**：
   ```python
   # 分析YOLO資料集
   stats = analyzer.analyze_yolo_dataset("path/to/your/data.yaml")
   
   # 獲取推薦
   recommendations = recommender.get_recommendations('detection', stats)
   ```

### 實際使用範例

#### 範例 1：使用 CIFAR-10 資料集

```bash
# 假設您有 CIFAR-10 資料集在 /data/cifar10/
python3 smart_recommendation_example.py --task classification --data_path /data/cifar10
```

#### 範例 2：使用 COCO 偵測資料集

```bash
# 假設您有 COCO 資料集在 /data/coco/
python3 smart_recommendation_example.py --task detection --data_path /data/coco/data.yaml
```

#### 範例 3：批量處理多個資料集

```bash
#!/bin/bash
# 批量處理腳本
datasets=(
    "/data/dataset1"
    "/data/dataset2" 
    "/data/dataset3"
)

for dataset in "${datasets[@]}"; do
    echo "處理資料集: $dataset"
    python3 smart_recommendation_example.py --task classification --data_path "$dataset"
    echo "---"
done
```

### 推薦結果說明

推薦結果包含以下資訊：

```python
{
    'model': 'efficientnet_b0',           # 推薦的模型
    'learning_rate': 0.001,               # 學習率
    'batch_size': 32,                     # 批次大小
    'epochs': 50,                         # 訓練輪數
    'optimizer': 'AdamW',                 # 優化器
    'data_augmentation': {...},           # 資料增強策略
    'regularization': {...},              # 正則化技術
    'reasoning': {                        # 推薦原因
        'model': 'EfficientNet provides good balance...',
        'learning_rate': 'Standard learning rate suitable...',
        'batch_size': 'Batch size 32 optimized for...',
        'epochs': 'Standard epoch count appropriate...'
    }
}
```

## 🛠️ 故障排除

### 常見問題

1. **找不到資料集**：
   - 確保資料集路徑正確
   - 檢查目錄結構是否符合要求

2. **依賴項缺失**：
   ```bash
   pip install opencv-python numpy pandas pyyaml pillow
   ```

3. **記憶體不足**：
   - 減少批次大小
   - 使用較小的模型
   - 降低圖片解析度

### 資料集格式要求

**分類資料集**：
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

**偵測資料集**：
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

## 📚 進階使用

### 自定義推薦邏輯

您可以擴展 `SmartRecommendationEngine` 類別來添加自定義的推薦邏輯：

```python
class CustomRecommendationEngine(SmartRecommendationEngine):
    def _recommend_classification_model(self, stats):
        # 您的自定義邏輯
        return super()._recommend_classification_model(stats)
```

### 整合到訓練流程

```python
from ivit.trainer.classification import ClassificationTrainer

# 獲取推薦
recommendations = recommender.get_recommendations('classification', stats)

# 創建訓練器
trainer = ClassificationTrainer(
    model_name=recommendations['model'],
    learning_rate=recommendations['learning_rate'],
    batch_size=recommendations['batch_size'],
    epochs=recommendations['epochs']
)

# 開始訓練
trainer.train()
```

## 🎉 總結

智能推薦系統讓您能夠：
- 自動分析資料集特性
- 獲得專業的參數推薦
- 節省調參時間
- 提高訓練效果

立即開始使用，讓AI模型訓練變得更智能！
