# iVIT SDK 開發路線圖

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

### Phase 1：API 簡化（學習 Ultralytics）

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

**待實作**：
- [ ] `ivit.load()` 統一載入函數
- [ ] 自動任務類型偵測（分類/偵測/分割）
- [ ] 支援直接傳入檔案路徑
- [ ] `results.show()` 快速視覺化
- [ ] `results.save()` 快速儲存

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

**待實作**：
- [ ] 統一的推論參數介面
- [ ] 串流模式（省記憶體）
- [ ] 批次推論支援
- [ ] 測試時增強 (TTA)

---

### Phase 2：進階控制

#### 2.1 Callback 系統（L3）

**目標**：提供擴展點，不改核心程式碼

```python
@model.on("pre_process")
def custom_preprocess(image, context):
    return my_preprocess(image)

@model.on("post_process")
def custom_postprocess(outputs, context):
    return my_postprocess(outputs)

@model.on("infer_end")
def log_metrics(results, context):
    send_to_prometheus(context["latency"])
```

**待實作**：
- [ ] Callback 註冊機制
- [ ] 支援的事件：
  - `pre_process` - 前處理前
  - `post_process` - 後處理後
  - `infer_start` - 推論開始
  - `infer_end` - 推論結束
  - `batch_start` - 批次開始
  - `batch_end` - 批次結束

#### 2.2 硬體特定配置（L4）

**目標**：暴露底層優化選項

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

# Qualcomm SNPE 特定設定
model.configure_snpe(
    runtime="dsp",
    performance_profile="high",
)
```

**待實作**：
- [ ] `configure_openvino()` 方法
- [ ] `configure_tensorrt()` 方法
- [ ] `configure_snpe()` 方法
- [ ] 配置驗證和錯誤提示

#### 2.3 底層存取（L5）

**目標**：專家可直接操作底層

```python
# 存取底層 runtime
runtime = model.runtime

# OpenVINO
ov_model = runtime.get_openvino_model()
ov_request = runtime.get_infer_request()

# TensorRT
engine = runtime.get_trt_engine()
context = runtime.get_trt_context()

# 自訂前/後處理
raw_output = model.infer_raw(tensor)
```

**待實作**：
- [ ] `model.runtime` 屬性
- [ ] `infer_raw()` 原始推論方法
- [ ] 底層物件存取方法

---

### Phase 3：開發體驗

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

#### 3.2 CLI 工具增強

**待實作**：
- [ ] `ivit info` - 顯示系統資訊
- [ ] `ivit devices` - 列出裝置
- [ ] `ivit benchmark` - 效能測試
- [ ] `ivit convert` - 模型轉換
- [ ] `ivit serve` - 啟動推論服務
- [ ] `ivit export` - 匯出模型

#### 3.3 錯誤訊息優化

**目標**：友善的錯誤提示

```python
# 現在
RuntimeError: Failed to load model

# 優化後
ivit.ModelLoadError: 無法載入模型 'model.onnx'

可能原因：
  1. 檔案不存在：請確認路徑 '/path/to/model.onnx' 是否正確
  2. 格式不支援：目前支援 .onnx, .xml, .engine
  3. 後端不可用：TensorRT 需要 NVIDIA GPU

解決建議：
  - 執行 `ivit.devices()` 確認可用硬體
  - 執行 `ivit convert model.onnx model.engine` 轉換格式
```

---

### Phase 4：生態建設

#### 4.1 Model Zoo

**目標**：預優化模型庫

```python
# 從 Model Zoo 載入
model = ivit.load("yolov8n")  # 自動下載 + 轉換
model = ivit.load("yolov8n", device="npu")  # 自動選擇 NPU 優化版

# 列出可用模型
ivit.zoo.list()
ivit.zoo.search("yolo")
```

#### 4.2 與 Ultralytics 整合

**目標**：無縫銜接 Ultralytics 工作流

```python
from ultralytics import YOLO

# 用 Ultralytics 訓練
model = YOLO("yolov8n.pt")
model.train(data="coco.yaml", epochs=100)
model.export(format="onnx")

# 用 iVIT 部署
import ivit
deployed = ivit.load("yolov8n.onnx", device=ivit.devices.npu())
results = deployed("image.jpg")
```

#### 4.3 文件和範例

**待完成**：
- [ ] API 參考文件
- [ ] 快速入門指南
- [ ] 各平台部署範例
- [ ] 效能調優指南
- [ ] 常見問題 FAQ

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
