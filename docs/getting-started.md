# iVIT-SDK Getting Started Guide

iVIT-SDK (Innodisk Vision Intelligence Toolkit) 是一個統一的電腦視覺推理 SDK，支援多種 AI 加速器後端，包括 Intel OpenVINO、NVIDIA TensorRT，以及 Qualcomm QNN (IQ Series)（規劃中）。

## 目錄

- [系統需求](#系統需求)
- [安裝](#安裝)
- [快速開始](#快速開始)
- [核心概念](#核心概念)
- [API 使用指南](#api-使用指南)
- [裝置選擇](#裝置選擇)
- [模型格式](#模型格式)
- [效能優化](#效能優化)
- [常見問題](#常見問題)

---

## 系統需求

### 作業系統
- Ubuntu 20.04 / 22.04 (推薦)
- Windows 10/11 (實驗性支援)

### 編譯器
- GCC 9+ 或 Clang 11+
- CMake 3.18+
- C++17 標準

### 依賴項
- OpenCV 4.5+
- (可選) OpenVINO 2024.0+
- (可選) CUDA 11.8+ 與 TensorRT >= 8.6
- (可選) Qualcomm AI Engine Direct SDK (QNN)（規劃中）

---

## 安裝

> **重要**：iVIT-SDK v1.0 起，Python SDK 需要 C++ 綁定。安裝時必須完成 C++ 編譯步驟。
> 純 Python runtime 已標記為 deprecated，將在未來版本移除。

### 方法一：從原始碼編譯

#### 前置套件安裝

```bash
# 安裝建置工具與依賴
sudo apt update
sudo apt install build-essential cmake pkg-config libopencv-dev

# 安裝 OpenVINO（若系統尚未安裝）
pip install openvino>=2024.0

# (選用) 安裝 pybind11（若需要 Python 綁定）
pip install pybind11
```

#### 編譯步驟

```bash
# 1. 取得原始碼
git clone https://github.com/innodisk-mannywang/ivit-sdk.git
cd ivit-sdk

# 2. (選用) 下載 C++ 後端依賴庫
#    若系統已安裝 OpenVINO (pip install openvino)，可跳過此步驟
#    CMake 會自動透過 find_package() 找到系統 OpenVINO
./scripts/download_deps.sh

# 3. 建立 build 目錄
mkdir build && cd build

# 4. 執行 CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_BUILD_EXAMPLES=ON \
    -DIVIT_BUILD_TESTS=OFF \
    -DIVIT_BUILD_PYTHON=OFF

# 5. 編譯
make -j$(nproc)

# 6. 安裝（可選）
sudo make install
```

> **重要**：
> - 若系統已透過 `pip install openvino` 安裝 OpenVINO，CMake 會自動找到，**不需要** `download_deps.sh` 也不需要 `-DIVIT_BUNDLE_DEPS=ON`。
> - `download_deps.sh` 適用於沒有系統級 OpenVINO 的環境，會下載獨立的 C++ Runtime 到 `deps/install/`，此時需搭配 `-DIVIT_BUNDLE_DEPS=ON`。
> - **TensorRT** 需要 **>= 8.6 版本**，請從 [NVIDIA Developer](https://developer.nvidia.com/tensorrt) 手動下載（需登入帳號），然後解壓到 `deps/install/tensorrt/`。

### download_deps.sh 選項

```bash
# 下載所有依賴（預設：OpenVINO）
./scripts/download_deps.sh

# 僅下載 OpenVINO
./scripts/download_deps.sh --openvino-only

# 顯示 TensorRT 手動安裝說明
./scripts/download_deps.sh --tensorrt

# 查看所有選項
./scripts/download_deps.sh --help
```

### 其他編譯配置

```bash
# 純 C++ 開發（不需要 Python 綁定）
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_BUILD_EXAMPLES=ON \
    -DIVIT_BUILD_TESTS=OFF \
    -DIVIT_BUILD_PYTHON=OFF \
    -DIVIT_BUNDLE_DEPS=ON

# 完整編譯（含測試，需確保系統 curl 版本相容）
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_BUILD_EXAMPLES=ON \
    -DIVIT_BUILD_TESTS=ON \
    -DIVIT_BUILD_PYTHON=ON \
    -DIVIT_BUNDLE_DEPS=ON
```

### CMake 選項

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `IVIT_BUILD_PYTHON` | ON | 編譯 Python bindings（需先安裝 pybind11） |
| `IVIT_BUILD_TESTS` | ON | 編譯單元測試 |
| `IVIT_BUILD_EXAMPLES` | ON | 編譯範例程式 |
| `IVIT_USE_OPENVINO` | ON | 啟用 OpenVINO 後端 |
| `IVIT_USE_TENSORRT` | ON | 啟用 TensorRT 後端 |
| `IVIT_BUNDLE_DEPS` | OFF | **使用 deps/install 中的 C++ SDK**（建議開啟） |

### 預期輸出

正確配置後，CMake 應顯示：

```
-- Backends:
--   OpenVINO:       ON
--   TensorRT:       ON
--   QNN:            OFF
```

若顯示 `OFF`，請確認：
1. 已設定 `-DIVIT_BUNDLE_DEPS=ON`
2. `deps/install/` 目錄存在且包含 SDK 檔案

### 方法二：使用預編譯套件

```bash
# 下載預編譯套件
wget https://github.com/innodisk-mannywang/ivit-sdk/releases/download/v1.0.0/ivit-sdk-1.0.0-linux-x64.tar.gz

# 解壓縮
tar -xzf ivit-sdk-1.0.0-linux-x64.tar.gz
cd ivit-sdk-1.0.0

# 設定環境變數
source bin/setup_env.sh
```

---

## 快速開始

### C++ 範例

#### 物件偵測

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ivit/vision/detector.hpp"

using namespace ivit;
using namespace ivit::vision;

int main() {
    // 載入模型
    Detector detector("yolov8n.onnx", "auto");

    // 讀取影像
    cv::Mat image = cv::imread("image.jpg");

    // 執行推理
    auto results = detector.predict(image, 0.5f);  // conf_threshold = 0.5

    // 處理結果
    for (const auto& det : results.detections) {
        std::cout << det.label << ": " << det.confidence * 100 << "%" << std::endl;

        // 繪製框框
        cv::rectangle(image, det.bbox.to_rect(), cv::Scalar(0, 255, 0), 2);
    }

    // 儲存結果
    cv::imwrite("output.jpg", image);

    return 0;
}
```

#### 影像分類

```cpp
#include "ivit/vision/classifier.hpp"

Classifier classifier("resnet50.onnx", "cpu");
auto results = classifier.predict(image, 5);  // Top-5

for (const auto& cls : results.classifications) {
    std::cout << cls.label << ": " << cls.score * 100 << "%" << std::endl;
}
```

#### 語意分割

```cpp
#include "ivit/vision/segmentor.hpp"

Segmentor segmentor("deeplabv3.onnx", "cuda:0");
auto results = segmentor.predict(image);

// 取得分割遮罩
cv::Mat mask = results.segmentation_mask;

// 視覺化
cv::Mat overlay = results.overlay_mask(image, 0.5);
cv::imwrite("segmentation.jpg", overlay);
```

### Python 範例

```python
import ivit

# 物件偵測
detector = ivit.Detector("yolov8n.onnx", device="cuda:0")
results = detector.predict("image.jpg")

for det in results.detections:
    print(f"{det.label}: {det.confidence:.2%}")
    print(f"  BBox: ({det.bbox.x1}, {det.bbox.y1}) - ({det.bbox.x2}, {det.bbox.y2})")

# 影像分類
classifier = ivit.Classifier("resnet50.onnx", device="cpu")
results = classifier.predict("image.jpg", top_k=5)

for cls in results.classifications:
    print(f"{cls.label}: {cls.score:.2%}")
```

---

## 核心概念

### 架構概覽

```
┌─────────────────────────────────────────────────────────────┐
│                      Application                             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Vision Tasks Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Classifier  │  │   Detector   │  │  Segmentor   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │    Model     │  │   Tensor     │  │   Results    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Runtime Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  OpenVINO    │  │  TensorRT    │  │     QNN      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Hardware Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Intel CPU   │  │ NVIDIA GPU   │  │ Qualcomm IQ  │       │
│  │  Intel GPU   │  │              │  │ (IQ9/IQ8/IQ6)│       │
│  │  Intel NPU   │  │              │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 主要類別

| 類別 | 說明 |
|------|------|
| `Detector` | 物件偵測（YOLO, SSD, Faster R-CNN） |
| `Classifier` | 影像分類（ResNet, EfficientNet） |
| `Segmentor` | 語意分割（DeepLabV3, U-Net） |
| `Model` | 底層模型抽象介面 |
| `Results` | 推理結果容器 |
| `DeviceManager` | 裝置管理與發現 |

---

## API 使用指南

### Detector API

```cpp
class Detector {
public:
    // 建構函式
    Detector(const std::string& model_path,
             const std::string& device = "auto",
             const LoadConfig& config = LoadConfig{});

    // 單張影像偵測
    Results predict(const cv::Mat& image,
                    float conf_threshold = 0.5f,
                    float iou_threshold = 0.45f);

    // 從檔案偵測
    Results predict(const std::string& image_path,
                    float conf_threshold = 0.5f,
                    float iou_threshold = 0.45f);

    // 批次偵測
    std::vector<Results> predict_batch(const std::vector<cv::Mat>& images,
                                       const InferConfig& config = InferConfig{});

    // 影片偵測
    void predict_video(const std::string& source,
                       std::function<void(const Results&, const cv::Mat&)> callback,
                       const InferConfig& config = InferConfig{});

    // 屬性
    const std::vector<std::string>& classes() const;
    int num_classes() const;
    cv::Size input_size() const;
};
```

### Results API

```cpp
class Results {
public:
    // 偵測結果
    std::vector<Detection> detections;

    // 分類結果
    std::vector<ClassificationResult> classifications;

    // 分割結果
    cv::Mat segmentation_mask;

    // 中繼資料
    float inference_time_ms;
    std::string device_used;
    cv::Size image_size;

    // 分類方法
    const ClassificationResult& top1() const;
    std::vector<ClassificationResult> topk(int k) const;

    // 偵測過濾
    std::vector<Detection> filter_by_class(const std::vector<int>& class_ids) const;
    std::vector<Detection> filter_by_confidence(float min_conf) const;

    // 分割視覺化
    cv::Mat colorize_mask(const std::map<int, cv::Vec3b>& colormap = {}) const;
    cv::Mat overlay_mask(const cv::Mat& image, float alpha = 0.5) const;

    // 通用視覺化
    cv::Mat visualize(const cv::Mat& image,
                      bool show_labels = true,
                      bool show_confidence = true,
                      bool show_boxes = true,
                      bool show_masks = true) const;

    // 匯出
    std::string to_json() const;
    void save(const std::string& path, const std::string& format = "json") const;
};
```

### Detection 結構

```cpp
struct Detection {
    BBox bbox;              // 邊界框 (x1, y1, x2, y2)
    int class_id;           // 類別 ID
    std::string label;      // 類別名稱
    float confidence;       // 信心分數 [0, 1]
    std::optional<cv::Mat> mask;  // 實例分割遮罩（可選）
};

struct BBox {
    float x1, y1, x2, y2;   // 左上角和右下角座標

    float width() const;
    float height() const;
    float area() const;
    float iou(const BBox& other) const;
    cv::Rect to_rect() const;
};
```

---

## 裝置選擇

### 支援的裝置

| 裝置字串 | 說明 | 後端 |
|----------|------|------|
| `auto` | 自動選擇最佳裝置 | 自動 |
| `cpu` | CPU | OpenVINO |
| `gpu:0` | Intel 內顯 | OpenVINO |
| `cuda:0` | NVIDIA GPU | TensorRT |
| `npu` | Intel NPU | OpenVINO |
| `vpu` | Intel VPU (Myriad) | OpenVINO |
| `iq9` | Qualcomm IQ9 (QCS9075) | QNN (規劃中) |
| `iq8` | Qualcomm IQ8 (QCS8550) | QNN (規劃中) |
| `iq6` | Qualcomm IQ6 (QCS6490) | QNN (規劃中) |

### 裝置發現

```cpp
#include "ivit/core/device.hpp"

auto& manager = DeviceManager::instance();
auto devices = manager.list_devices();

for (const auto& dev : devices) {
    std::cout << "Device: " << dev.id << std::endl;
    std::cout << "  Name:    " << dev.name << std::endl;
    std::cout << "  Backend: " << dev.backend << std::endl;
    std::cout << "  Type:    " << dev.type << std::endl;
}

// 取得最佳裝置
auto best = manager.get_best_device("detection", "performance");
std::cout << "Best device: " << best.id << std::endl;
```

### 裝置選擇策略

```cpp
// 明確指定裝置
Detector detector("model.onnx", "cuda:0");

// 自動選擇（優先 GPU）
Detector detector("model.onnx", "auto");

// 指定後端
LoadConfig config;
config.backend = "tensorrt";
config.device = "cuda:0";
Detector detector("model.onnx", config.device, config);
```

---

## 模型格式

### 支援的格式

| 格式 | 副檔名 | 後端 | 說明 |
|------|--------|------|------|
| ONNX | `.onnx` | 全部 | 通用交換格式（推薦） |
| OpenVINO IR | `.xml`, `.bin` | OpenVINO | Intel 優化格式 |
| TensorRT Engine | `.engine`, `.trt` | TensorRT | NVIDIA 優化格式 |
| PaddlePaddle | `.pdmodel` | OpenVINO | 百度框架格式 |

### 模型轉換

```cpp
#include "ivit/core/model.hpp"

auto& manager = ModelManager::instance();

// ONNX 轉 OpenVINO IR
manager.convert("model.onnx", "model.xml", BackendType::OpenVINO, Precision::FP16);

// ONNX 轉 TensorRT Engine
manager.convert("model.onnx", "model.engine", BackendType::TensorRT, Precision::FP16);
```

### 精度設定

| 精度 | 說明 | 適用場景 |
|------|------|----------|
| `FP32` | 32 位元浮點 | 最高精度 |
| `FP16` | 16 位元浮點 | 平衡精度與效能 |
| `INT8` | 8 位元整數 | 最高效能（需校準） |

```cpp
LoadConfig config;
config.precision = "fp16";  // 使用 FP16
Detector detector("model.onnx", "cuda:0", config);
```

---

## 效能優化

### 1. 選擇正確的後端

```
效能排序（相同硬體）:
TensorRT (NVIDIA GPU) > OpenVINO (Intel CPU/GPU/NPU)
```

### 2. 使用 FP16 精度

```cpp
LoadConfig config;
config.precision = "fp16";
Detector detector("model.onnx", "cuda:0", config);
```

### 3. 模型預熱

```cpp
// 預熱 3 次，避免首次推理延遲
detector.model()->warmup(3);
```

### 4. 啟用模型快取

```cpp
LoadConfig config;
config.use_cache = true;
config.cache_dir = "/path/to/cache";
Detector detector("model.onnx", "cuda:0", config);
```

### 5. 效能測試

```bash
# 使用範例程式進行效能測試
./simple_inference benchmark yolov8n.onnx cuda:0 100
```

預期輸出：
```
========================================
 Benchmark
========================================
Model:      yolov8n.onnx
Device:     cuda:0
Iterations: 100

Warming up...
Running benchmark...

Results:
--------
  Total time:  1234 ms
  Average:     12.34 ms
  Min:         11.20 ms
  Max:         15.60 ms
  P50:         12.10 ms
  P95:         14.50 ms
  P99:         15.20 ms
  Throughput:  81.04 FPS
```

---

## 常見問題

### Q: 如何檢查後端是否可用？

```cpp
#include "ivit/core/device.hpp"

if (openvino_is_available()) {
    std::cout << "OpenVINO is available" << std::endl;
}

if (cuda_is_available()) {
    std::cout << "CUDA is available (" << cuda_device_count() << " devices)" << std::endl;
}
```

### Q: 模型載入失敗怎麼辦？

1. 確認模型檔案存在且格式正確
2. 確認對應的後端已安裝
3. 檢查錯誤訊息

```cpp
try {
    Detector detector("model.onnx", "cuda:0");
} catch (const ModelLoadError& e) {
    std::cerr << "Model load failed: " << e.what() << std::endl;
} catch (const DeviceNotFoundError& e) {
    std::cerr << "Device not found: " << e.what() << std::endl;
}
```

### Q: 如何處理記憶體不足？

```cpp
// 使用較小的模型
Detector detector("yolov8n.onnx", "cuda:0");  // nano 版本

// 或降低精度
LoadConfig config;
config.precision = "fp16";  // 減少一半記憶體
Detector detector("yolov8s.onnx", "cuda:0", config);
```

### Q: 如何支援自訂類別？

在模型目錄下建立 `labels.txt`：

```
# labels.txt
person
car
bicycle
...
```

或在程式中載入：

```cpp
// 模型會自動讀取同目錄下的 labels.txt
Detector detector("my_model/model.onnx", "auto");
std::cout << "Classes: " << detector.num_classes() << std::endl;
```

---

## 範例程式

SDK 包含多個範例程式：

### C++ 範例

| 範例 | 說明 |
|------|------|
| `simple_inference` | 綜合範例（分類、偵測、分割、效能測試） |
| `classification_demo` | 影像分類範例 |
| `detection_demo` | 物件偵測範例 |
| `video_demo` | 即時影片偵測範例 |

#### 執行 C++ 範例

```bash
# 列出可用裝置
./simple_inference devices

# 物件偵測
./simple_inference detect yolov8n.onnx image.jpg cuda:0 output.jpg

# 影像分類
./simple_inference classify resnet50.onnx cat.jpg cpu

# 語意分割
./simple_inference segment deeplabv3.onnx scene.jpg gpu:0 segmented.jpg

# 效能測試
./simple_inference benchmark yolov8n.onnx cuda:0 100
```

### Python 範例

| 範例 | 說明 |
|------|------|
| `01_quickstart.py` | 快速入門範例（直接執行即可） |
| `02_detection.py` | 物件偵測完整範例（支援參數設定、效能測試） |
| `02_classification.py` | 影像分類範例 |

#### 執行 Python 範例

```bash
# 快速入門（自動選擇最佳裝置）
python examples/python/01_quickstart.py

# 物件偵測（指定裝置）
python examples/python/02_detection.py \
    -m models/onnx/yolov8n.onnx \
    -i examples/data/images/bus.jpg \
    -d cuda:0

# 物件偵測（效能測試模式）
python examples/python/02_detection.py \
    -m models/onnx/yolov8n.onnx \
    -i examples/data/images/bus.jpg \
    -d cuda:0 \
    -b -n 100

# 影像分類
python examples/python/02_classification.py \
    -m models/onnx/resnet50.onnx \
    -i examples/data/images/cat.jpg \
    -d cpu
```

#### 常用參數說明

| 參數 | 說明 |
|------|------|
| `-m, --model` | 模型檔案路徑（.onnx, .engine, .xml） |
| `-i, --image` | 輸入圖片路徑 |
| `-s, --source` | 影片來源（攝影機編號或影片路徑） |
| `-d, --device` | 推論裝置（cuda:0, gpu:0, cpu, auto） |
| `-o, --output` | 輸出檔案路徑 |
| `-b, --benchmark` | 啟用效能測試模式 |
| `-n, --iterations` | 效能測試迭代次數 |
| `--conf` | 信心度閾值（預設 0.25） |
| `--iou` | NMS IoU 閾值（預設 0.45） |

---

## 聯絡資訊

- **GitHub**: https://github.com/innodisk-mannywang/ivit-sdk
- **Issues**: https://github.com/innodisk-mannywang/ivit-sdk/issues
- **Email**: support@innodisk.com

---

*iVIT-SDK v1.0.0 - Innodisk Corporation*
