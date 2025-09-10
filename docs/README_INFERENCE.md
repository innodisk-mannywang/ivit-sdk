# iVIT 2.0 SDK - 推理使用指南

本指南介紹如何使用 iVIT 2.0 SDK 的推理模組進行模型推理，支援分類、偵測、分割三大任務。

## 📋 支援的推理類型

- **Classification** - 圖像分類推理
- **Detection** - 物件偵測推理  
- **Segmentation** - 語義分割推理
- **Unified** - 統一推理接口

## 🚀 快速開始

### 1. 分類推理

> **💡 自動類別名稱載入**
> 
> 當使用自定義模型時，推理器會自動尋找對應的 `class_names.json` 檔案。
> 檔案命名規則：`{模型名稱}_class_names.json`
> 
> 例如：如果模型檔案是 `classification_resnet18_20241201_143022.pth`，
> 系統會自動尋找 `classification_resnet18_20241201_143022_class_names.json`

```bash
# 手動指定類別名稱檔案
python examples/inference/classification_inference_example.py \
    --image /path/to/image.jpg \
    --model_path /path/to/model.pth \
    --num_classes 10 \
    --class_names /path/to/class_names.json
```

### 2. 偵測推理

```bash
# 使用自定義模型和參數
python examples/inference/detection_inference_example.py \
    --image /path/to/image.jpg \
    --model_path /path/to/yolo_model.pt \
    --conf_threshold 0.3 \
    --iou_threshold 0.5 \
    --draw_boxes \
    --save_image
```

### 3. 分割推理

```bash
# 使用自定義模型和參數
python examples/inference/segmentation_inference_example.py \
    --image /path/to/image.jpg \
    --model_path /path/to/seg_model.pt \
    --num_classes 21 \
    --conf_threshold 0.3 \
    --save_masks \
    --save_overlay
```

### 4. 統一推理

```bash
# 單任務推理
python examples/inference/unified_inference_example.py \
    --image /path/to/image.jpg \
    --models classification

# 多任務推理
python examples/inference/unified_inference_example.py \
    --image /path/to/image.jpg \
    --models classification detection segmentation

# 使用自定義模型
python examples/inference/unified_inference_example.py \
    --image /path/to/image.jpg \
    --models classification detection \
    --model_paths /path/to/class_model.pth /path/to/det_model.pt \
    --save_results \
    --benchmark
```

## 🔧 程式碼使用方式

### 分類推理

```python
from ivit.inference import ClassificationInference

# 創建推理器
inference = ClassificationInference(
    model_name='resnet18',
    num_classes=1000,
    device='cuda'
)

# 載入模型
inference.load_model('/path/to/model.pth')

# 執行推理
results = inference.predict('/path/to/image.jpg')
print(f"預測類別: {results['top_class']}")
print(f"信心度: {results['top_confidence']:.4f}")

# 獲取前5個預測
top5 = inference.predict_top_k('/path/to/image.jpg', k=5)
for pred in top5['predictions']:
    print(f"{pred['class_name']}: {pred['confidence']:.4f}")
```

### 偵測推理

```python
from ivit.inference import DetectionInference

# 創建推理器
inference = DetectionInference(
    model_name='yolov8n.pt',
    conf_threshold=0.25,
    device='cuda'
)

# 載入模型
inference.load_model('/path/to/yolo_model.pt')

# 執行推理
results = inference.predict('/path/to/image.jpg')
print(f"偵測到 {results['num_detections']} 個物件")

# 繪製偵測框
output_image = inference.draw_detections('/path/to/image.jpg', results)
cv2.imwrite('result.jpg', output_image)

# 過濾偵測結果
filtered = inference.filter_by_confidence(results, min_confidence=0.5)
print(f"高信心度偵測: {filtered['num_detections']} 個")
```

### 分割推理

```python
from ivit.inference import SegmentationInference

# 創建推理器
inference = SegmentationInference(
    model_name='yolov8n-seg.pt',
    num_classes=21,
    device='cuda'
)

# 載入模型
inference.load_model('/path/to/seg_model.pt')

# 執行推理
results = inference.predict('/path/to/image.jpg')
print(f"偵測到 {results['num_masks']} 個遮罩")

# 創建彩色遮罩疊加
overlay = inference.create_colored_mask('/path/to/image.jpg', results['masks'])
cv2.imwrite('overlay.jpg', overlay)

# 創建類別遮罩
class_mask = inference.create_class_mask(results['image_shape'], results['masks'])
cv2.imwrite('class_mask.png', class_mask)
```

### 統一推理

```python
from ivit.inference import UnifiedInference

# 創建統一推理器
inference = UnifiedInference(device='cuda')

# 載入多個模型
inference.load_model('classification', '/path/to/class_model.pth')
inference.load_model('detection', '/path/to/det_model.pt')
inference.load_model('segmentation', '/path/to/seg_model.pt')

# 單任務推理
class_results = inference.predict_classification('/path/to/image.jpg')
det_results = inference.predict_detection('/path/to/image.jpg')

# 多任務推理
multi_results = inference.predict_multi_task('/path/to/image.jpg', 
                                           ['classification', 'detection'])

# 效能測試
benchmark = inference.benchmark_all_models('/path/to/image.jpg')
```

## 📊 支援的模型格式

### 分類模型
- **PyTorch 模型** (.pth, .pt)
- **支援架構**: ResNet, EfficientNet, MobileNet, DenseNet
- **預訓練模型**: ImageNet 預訓練權重

### 偵測模型
- **YOLO 模型** (.pt)
- **支援架構**: YOLOv8 系列 (n, s, m, l, x)
- **預訓練模型**: COCO 預訓練權重

### 分割模型
- **YOLO-seg 模型** (.pt)
- **支援架構**: YOLOv8-seg 系列 (n, s, m, l, x)
- **預訓練模型**: COCO 預訓練權重

## 🎯 進階功能

### 批次推理

```python
# 批次推理
images = ['/path/to/img1.jpg', '/path/to/img2.jpg', '/path/to/img3.jpg']
results = inference.predict_batch(images)

# 資料夾推理
folder_results = inference.predict_from_folder('/path/to/images/')
```

### 效能優化

```python
# 效能測試
benchmark = inference.benchmark('/path/to/image.jpg', num_runs=100)
print(f"平均推理時間: {benchmark['mean_time']:.4f} 秒")
print(f"推理速度: {benchmark['fps']:.2f} FPS")

# 模型資訊
model_info = inference.get_model_info()
print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
```

### 結果後處理

```python
# 分類結果過濾
filtered = inference.predict_with_confidence_threshold(
    '/path/to/image.jpg', threshold=0.8
)

# 偵測結果過濾
filtered_det = inference.filter_by_confidence(det_results, min_confidence=0.7)
filtered_size = inference.filter_by_size(det_results, min_area=1000)

# 分割結果過濾
filtered_masks = inference.filter_masks_by_class(seg_results['masks'], [0, 1, 2])
```

## 🛠️ 故障排除

### 常見問題

1. **模型載入失敗**
   - 檢查模型檔案路徑
   - 確認模型格式是否支援
   - 檢查類別數量是否匹配

2. **記憶體不足**
   - 減少批次大小
   - 使用較小的模型
   - 切換到 CPU 推理

3. **推理速度慢**
   - 使用 GPU 推理
   - 選擇較小的模型
   - 減少輸入圖像尺寸

4. **結果不準確**
   - 檢查預處理參數
   - 確認模型與資料集匹配
   - 調整信心度閾值

### 調試模式

```python
# 啟用詳細日誌
import logging
logging.basicConfig(level=logging.DEBUG)

# 檢查模型載入
model_info = inference.get_model_info()
print(f"模型資訊: {model_info}")

# 測試單張圖像
try:
    result = inference.predict('/path/to/test_image.jpg')
    print("推理成功")
except Exception as e:
    print(f"推理失敗: {e}")
```

## 📚 更多範例

查看 `examples/inference/` 目錄中的完整範例：

- `classification_inference_example.py` - 分類推理範例
- `detection_inference_example.py` - 偵測推理範例
- `segmentation_inference_example.py` - 分割推理範例
- `unified_inference_example.py` - 統一推理範例

## 🤝 支援

如有問題，請參考：
- 專案文檔
- 範例代碼
- 問題回報

---

**注意**: 請確保已正確安裝所有依賴項，並配置好 CUDA 環境（如使用 GPU 推理）。
