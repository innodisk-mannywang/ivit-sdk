# iVIT SDK 開發路線圖

> **更新日期**：2026-01-26

## 開發進度總覽

| Phase | 狀態 | 完成度 |
|-------|------|--------|
| Phase 1: API 簡化 | ✅ 完成 | 100% |
| Phase 2: 進階控制 | ✅ 完成 | 100% |
| Phase 3: 開發體驗 | ✅ 完成 | 100% |
| Phase 4: 生態建設 | ✅ 完成 | 100% |

---

## 定位策略

### 與競品的差異化

| SDK | 定位 | iVIT 的機會 |
|-----|------|------------|
| **Ultralytics** | YOLO 訓練 + 推論 | iVIT 專注部署，不做訓練 |
| **OpenVINO** | Intel 硬體專用 | iVIT 跨平台統一 |
| **TensorRT** | NVIDIA 硬體專用 | iVIT 跨平台統一 |

**核心價值主張**：「用任何框架訓練，用 iVIT 部署到任何硬體」

---

## 功能優化路線圖

### Phase 1：API 簡化（學習 Ultralytics）✅

#### 1.1 極簡 API（L1）

**目標**：一行載入、一行推論

```python
# 現在
from ivit.vision import Detector
detector = Detector("yolov8n.onnx", device="cuda:0")
results = detector.predict(image)

# 優化後
import ivit
model = ivit.load("yolov8n.onnx")
results = model("image.jpg")  # 自動偵測任務類型
results.show()
```

**已完成** ✅：
- [x] `ivit.load()` 統一載入函數
- [x] 自動任務類型偵測（分類/偵測/分割）
- [x] 支援直接傳入檔案路徑
- [x] `results.show()` 快速視覺化
- [x] `results.save()` 快速儲存
- [x] `model()` 直接呼叫推論（Ultralytics 風格）

#### 1.2 參數配置（L2）

**目標**：豐富的推論參數

```python
results = model.predict(
    source="image.jpg",
    conf=0.25,           # 信心度閾值
    iou=0.45,            # NMS IoU 閾值
    max_det=300,         # 最大偵測數
    classes=[0, 1, 2],   # 只偵測特定類別
    imgsz=640,           # 輸入尺寸
    half=True,           # FP16 推論
    augment=True,        # 測試時增強
    save=True,           # 儲存結果
    stream=True,         # 串流模式
)
```

**已完成** ✅：
- [x] 統一的推論參數介面 (`conf`, `iou`, `classes`, `max_det`)
- [x] 串流模式 `model.stream()` generator 模式
- [x] 批次推論支援 `model.predict_batch()`
- [x] 測試時增強 `model.predict_tta()` (TTA)

**TTA 使用範例**：
```python
# 基本 TTA（水平翻轉）
results = model.predict_tta("image.jpg")

# 多尺度 TTA
results = model.predict_tta("image.jpg", scales=[0.8, 1.0, 1.2])

# 自訂增強組合
results = model.predict_tta(
    "image.jpg",
    augments=["original", "hflip", "rotate90"],
    scales=[0.8, 1.0, 1.2]
)
```

---

### Phase 2：進階控制 ✅

#### 2.1 Callback 系統（L3）✅

**已完成**：提供擴展點，不改核心程式碼

```python
# 使用裝飾器註冊
@model.on("infer_end")
def log_metrics(ctx):
    print(f"Latency: {ctx.latency_ms}ms")
    print(f"Preprocess: {ctx.preprocess_ms}ms")
    print(f"Inference: {ctx.inference_ms}ms")

# 使用 lambda
model.on("pre_process", lambda ctx: print(f"Image shape: {ctx.image_shape}"))

# 內建 callbacks
from ivit.core.callbacks import FPSCounter, LatencyLogger, DetectionFilter
fps = FPSCounter(window_size=30)
model.on("infer_end", fps)
print(fps.fps)  # Get current FPS
```

**已實作**：
- [x] `model.on()` Callback 註冊（裝飾器 + 直接呼叫）
- [x] `model.remove_callback()` 移除 callback
- [x] `CallbackContext` 包含完整推論資訊
- [x] 支援的事件：
  - `pre_process` - 前處理前
  - `post_process` - 後處理後
  - `infer_start` - 推論開始
  - `infer_end` - 推論結束（含詳細時間）
  - `batch_start/batch_end` - 批次處理
  - `stream_start/frame/end` - 串流處理
- [x] 內建 callbacks：`FPSCounter`, `LatencyLogger`, `DetectionFilter`

#### 2.2 硬體特定配置（L4）✅

**已完成**：暴露底層優化選項

```python
# OpenVINO 特定設定
model.configure_openvino(
    performance_mode="LATENCY",
    num_streams=4,
    inference_precision="FP16",
    enable_cpu_pinning=True,
)

# TensorRT 特定設定
model.configure_tensorrt(
    workspace_size=1 << 30,
    dla_core=0,
    enable_sparsity=True,
    builder_optimization_level=3,
)

# ONNX Runtime 設定
model.configure_onnxruntime(
    num_threads=8,
    enable_cuda_graph=True,
)

# 支援 method chaining
model.configure_onnxruntime(num_threads=4).warmup(3)
```

**已實作**：
- [x] `configure_openvino()` 方法
- [x] `configure_tensorrt()` 方法
- [x] `configure_onnxruntime()` 方法
- [x] `configure_snpe()` 方法
- [x] 配置類別：`OpenVINOConfig`, `TensorRTConfig`, `ONNXRuntimeConfig`, `SNPEConfig`

#### 2.3 底層存取（L5）✅

**已完成**：專家可直接操作底層

```python
# 存取 runtime
runtime = model.runtime
print(runtime.name)  # "onnxruntime"

# 存取底層 handle
handle = model.runtime_handle
session = handle.session  # ONNX InferenceSession

# 原始推論（無前後處理）
input_name = model.input_info[0]["name"]
outputs = model.infer_raw({input_name: my_tensor})
```

**已實作**：
- [x] `model.runtime` 屬性
- [x] `model.runtime_handle` 屬性
- [x] `model.infer_raw()` 原始推論方法

---

### Phase 3：開發體驗 ✅

#### 3.1 裝置管理（已完成 ✅）

```python
# 列出裝置
ivit.devices()
ivit.devices.summary()

# 取得特定裝置
ivit.devices.cuda()
ivit.devices.cpu()
ivit.devices.npu()
ivit.devices.best()
ivit.devices.best("efficiency")
```

#### 3.2 CLI 工具增強 ✅

**已完成**：
- [x] `ivit info` - 顯示系統資訊
- [x] `ivit devices` - 列出裝置
- [x] `ivit benchmark` - 效能測試
- [x] `ivit predict` - 執行推論
- [x] `ivit convert` - 模型轉換（OpenVINO、TensorRT）
- [x] `ivit serve` - 啟動推論服務
- [x] `ivit export` - 匯出模型
- [x] `ivit zoo` - Model Zoo 操作

#### 3.3 錯誤訊息優化 ✅

**已完成**：友善的錯誤提示

```python
# 優化後的錯誤訊息範例
ivit.ModelLoadError: 無法載入模型 'model.onnx'

可能原因：
  1. 檔案不存在：請確認路徑 '/path/to/model.onnx' 是否正確
  2. 格式不支援：目前支援 .onnx, .xml, .engine
  3. 後端不可用：TensorRT 需要 NVIDIA GPU

解決建議：
  - 執行 `ivit.devices()` 確認可用硬體
  - 執行 `ivit convert model.onnx model.engine` 轉換格式
```

**已實作**：
- [x] `IVITError` 基礎錯誤類別
- [x] `ModelLoadError` 模型載入錯誤
- [x] `DeviceNotFoundError` 裝置不可用
- [x] `BackendNotAvailableError` 後端未安裝
- [x] `InferenceError` 推論錯誤
- [x] `InvalidInputError` 輸入格式錯誤
- [x] `ConfigurationError` 配置錯誤
- [x] `ModelConversionError` 模型轉換錯誤
- [x] `ResourceExhaustedError` 資源不足
- [x] `wrap_error()` 包裝通用錯誤

---

### Phase 4：生態建設 ✅

#### 4.1 Model Zoo ✅

**已完成**：

```python
# 從 Model Zoo 載入
model = ivit.zoo.load("yolov8n")  # 自動下載 + 轉換
model = ivit.zoo.load("yolov8n", device="cuda:0")

# 列出可用模型
ivit.zoo.list_models()              # 14 個預設模型
ivit.zoo.list_models(task="detect") # 按任務過濾
ivit.zoo.search("yolo")             # 搜尋模型
ivit.zoo.search("edge")             # 找邊緣裝置模型

# 取得模型資訊
info = ivit.zoo.get_model_info("yolov8n")
print(f"Task: {info.task}, Input: {info.input_size}")
```

**已支援模型**：
| 類型 | 模型 |
|------|------|
| Detection | yolov8n/s/m/l/x |
| Classification | yolov8n-cls, yolov8s-cls, resnet50, mobilenetv3, efficientnet-b0 |
| Segmentation | yolov8n-seg, yolov8s-seg |
| Pose | yolov8n-pose, yolov8s-pose |

#### 4.2 模型訓練 ✅

**已完成**：自行實作遷移式學習訓練功能

```python
import ivit
from ivit.train import Trainer, ImageFolderDataset, EarlyStopping, ModelCheckpoint

# 建立訓練資料集
dataset = ImageFolderDataset("./my_dataset", train_split=0.8, split="train")
val_dataset = ImageFolderDataset("./my_dataset", train_split=0.8, split="val")

# 訓練分類模型
trainer = Trainer(
    model="resnet50",           # 支援 resnet18/34/50/101, efficientnet_b0-b2, mobilenet_v2/v3
    dataset=dataset,
    val_dataset=val_dataset,
    epochs=20,
    learning_rate=0.001,
    batch_size=32,
    device="cuda:0",
    freeze_backbone=True,       # 遷移式學習：凍結骨幹網路
)

# 訓練並使用回調
trainer.fit(callbacks=[
    EarlyStopping(patience=5),
    ModelCheckpoint("best_model.pt"),
])

# 評估模型
metrics = trainer.evaluate()
print(f"Accuracy: {metrics['accuracy']:.2%}")

# 匯出模型
trainer.export("my_model.onnx", format="onnx", quantize="fp16")
```

**已實作**：
- [x] `ivit.train.ImageFolderDataset` - ImageFolder 格式資料集
- [x] `ivit.train.COCODataset` - COCO 格式資料集
- [x] `ivit.train.YOLODataset` - YOLO 格式資料集
- [x] `ivit.train.Trainer` - 訓練器（遷移式學習）
- [x] `ivit.train.Augmentation` - 資料增強（Resize, Flip, Rotation, ColorJitter, Normalize）
- [x] 訓練回調：EarlyStopping, ModelCheckpoint, ProgressLogger, LRScheduler, TensorBoardLogger
- [x] 模型匯出：ONNX, TorchScript, OpenVINO IR, TensorRT Engine
- [x] 量化支援：FP16, INT8（需要校正資料）
- [x] 支援 14+ 預訓練模型（ResNet, EfficientNet, MobileNet, VGG, DenseNet）

#### 4.3 文件和範例 ✅

**已完成**：
- [x] API 參考文件 (`docs/api/API-SPEC-001-ivit-sdk.md` v1.1)
- [x] 快速入門指南 (`docs/GETTING_STARTED.md`)
- [x] 各平台部署範例：
  - [x] Intel OpenVINO (`docs/deployment/INTEL_OPENVINO.md`)
  - [x] NVIDIA TensorRT (`docs/deployment/NVIDIA_TENSORRT.md`)
  - [x] Qualcomm SNPE (`docs/deployment/QUALCOMM_SNPE.md`)
- [x] 效能調優指南 (`docs/PERFORMANCE_TUNING.md`)
- [x] 常見問題 FAQ (`docs/FAQ.md`)

**已完成** ✅：
- [x] 訓練功能教學（ivit.train 模組）

---

## TODO: 待修復問題

### 🔴 高優先級：推論引擎優先順序修正

**問題描述**：目前 SDK 在 NVIDIA GPU 上預設使用 ONNX Runtime + CUDAExecutionProvider，而非效能更佳的 TensorRT。

**根因分析**：
```
get_backend_for_device("cuda:0")
    ↓
_tensorrt_available() 檢查失敗（需要 tensorrt + pycuda）
    ↓
回退到 _onnxruntime_cuda_available()
    ↓
使用 ONNX Runtime（效能較差）
```

**期望行為**：
| 硬體平台 | 第一優先 | 第二優先（回退） |
|----------|----------|------------------|
| NVIDIA GPU | TensorRT | ONNX Runtime + CUDA |
| Intel CPU/iGPU/NPU | OpenVINO | ONNX Runtime |
| Qualcomm NPU | SNPE/QNN | ONNX Runtime |
| 通用 CPU | OpenVINO | ONNX Runtime |

**修改位置**：
- `python/ivit/core/device.py`: `get_backend_for_device()` 函數
- `python/ivit/devices.py`: 裝置探索時標記實際可用的後端
- `python/ivit/runtime/__init__.py`: 增加更細緻的後端可用性檢查

**預估效能提升**：
- TensorRT vs ONNX Runtime + CUDA: 2-3x 速度提升
- OpenVINO vs ONNX Runtime (CPU): 1.5-2x 速度提升

---

## 效能優化

### SDK 包裝層開銷目標

| 指標 | 目前 | 目標 |
|------|------|------|
| Python 呼叫開銷 | ~0.3ms | <0.1ms |
| C++ 呼叫開銷 | ~0.05ms | <0.01ms |
| 記憶體額外開銷 | ~10MB | <5MB |

### 優化方向

- [ ] 減少 Python/C++ 邊界的資料複製
- [ ] 使用 buffer protocol 零複製傳輸
- [ ] 預分配輸出 buffer
- [ ] 批次推論優化

---

## 時程規劃

| Phase | 內容 | 預估時間 |
|-------|------|---------|
| Phase 1 | API 簡化 | 4 週 |
| Phase 2 | 進階控制 | 4 週 |
| Phase 3 | 開發體驗 | 4 週 |
| Phase 4 | 生態建設 | 持續 |

---

## 參考資料

- [Ultralytics API 設計](https://docs.ultralytics.com/)
- [OpenVINO Python API](https://docs.openvino.ai/latest/api/ie_python_api.html)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
