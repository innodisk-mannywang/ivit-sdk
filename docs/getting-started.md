# iVIT-SDK Getting Started Guide

iVIT-SDK (Innodisk Vision Intelligence Toolkit) 是一個統一的電腦視覺推理 SDK，支援多種 AI 加速器後端，包括 Intel OpenVINO、NVIDIA TensorRT，以及 Qualcomm QNN (IQ Series)（規劃中）。

## 目錄

- [系統需求](#系統需求)
- [環境安裝](#環境安裝)
- [編譯 SDK](#編譯-sdk)
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
- Ubuntu 22.04（推薦）

### 硬體支援
| 廠商 | 硬體 | 後端 |
|------|------|------|
| Intel | CPU / iGPU / NPU | OpenVINO |
| NVIDIA | dGPU / Jetson | TensorRT |
| Qualcomm | IQ9 / IQ8 / IQ6 | QNN（規劃中）|

### 軟體依賴

| 套件 | 用途 | 安裝方式 |
|------|------|----------|
| build-essential | C++ 編譯器 (GCC 11+) | APT |
| cmake | 建置系統 (3.18+) | APT |
| pkg-config | 套件偵測 | APT |
| libopencv-dev | OpenCV 4.5+ (含 dnn 模組) | APT |
| libopenvino-dev | OpenVINO C++ Runtime | Intel APT |
| python3-dev | Python 標頭檔（Python 綁定需要）| APT |
| pybind11 | Python/C++ 綁定（Python 綁定需要）| pip |

> **重要**：OpenVINO 必須使用 **APT 安裝的 C++ 版本**，不能用 `pip install openvino`。
> pip 版本的 OpenVINO 會造成 C++ ABI 不相容（`_GLIBCXX_USE_CXX11_ABI=0`），導致與系統 OpenCV 連結失敗。
> pip 版本僅適用於純 Python 開發，不適合 C++ 編譯。

---

## 環境安裝

### Step 1：安裝建置工具

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libopencv-dev
```

### Step 2：安裝 OpenVINO（Intel APT 來源）

```bash
# 加入 Intel APT 金鑰
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

# 加入 OpenVINO APT 來源
sudo bash -c 'echo "deb https://apt.repos.intel.com/openvino/2025 ubuntu22 main" > /etc/apt/sources.list.d/intel-openvino-2025.list'

# 安裝 OpenVINO C++ Runtime + 開發標頭 + 裝置 Plugin
sudo apt update
sudo apt install \
    libopenvino-dev-2025.4.1 \
    libopenvino-intel-cpu-plugin-2025.4.1 \
    libopenvino-intel-npu-plugin-2025.4.1 \
    libopenvino-onnx-frontend-2025.4.1
```

> **NPU 支援**：`libopenvino-intel-npu-plugin` 提供 Intel NPU 推論支援。
> 若不需要 NPU，可以省略該套件。

### Step 3：（選用）安裝 Python 綁定依賴

若需要編譯 Python binding（`-DIVIT_BUILD_PYTHON=ON`）：

```bash
sudo apt install python3-dev
pip install pybind11
```

### 驗證安裝

```bash
# 確認 cmake 版本
cmake --version

# 確認 OpenCV
pkg-config --modversion opencv4

# 確認 OpenVINO CMake 可被找到
ls /usr/lib/x86_64-linux-gnu/cmake/openvino/
```

---

## 編譯 SDK

### 取得原始碼

```bash
git clone https://github.com/innodisk-mannywang/ivit-sdk.git
cd ivit-sdk
```

### C++ 編譯（不含 Python 綁定）

```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_BUILD_EXAMPLES=ON \
    -DIVIT_BUILD_TESTS=OFF \
    -DIVIT_BUILD_PYTHON=OFF
make -j$(nproc)
```

### C++ + Python 綁定編譯

```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_BUILD_EXAMPLES=ON \
    -DIVIT_BUILD_TESTS=OFF \
    -DIVIT_BUILD_PYTHON=ON
make -j$(nproc)
```

### CMake 選項

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `IVIT_BUILD_PYTHON` | ON | 編譯 Python bindings |
| `IVIT_BUILD_TESTS` | ON | 編譯單元測試 |
| `IVIT_BUILD_EXAMPLES` | ON | 編譯範例程式 |
| `IVIT_USE_OPENVINO` | ON | 啟用 OpenVINO 後端 |
| `IVIT_USE_TENSORRT` | ON | 啟用 TensorRT 後端 |

### 預期輸出

CMake 正確配置後應顯示：

```
-- OpenCV version: 4.5.4
-- OpenVINO found: 2025.4.1
--
-- Backends:
--   OpenVINO:       ON
--   TensorRT:       OFF
--   QNN:            OFF
```

### 編譯產物

```bash
ls build/lib/          # libivit.so
ls build/bin/          # simple_inference, detection_demo, ...
```

---

## 快速開始

### 確認裝置

```bash
cd build
./bin/simple_inference devices
```

### C++ 物件偵測

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ivit/vision/detector.hpp"

using namespace ivit;
using namespace ivit::vision;

int main() {
    // 載入模型（使用 NPU）
    Detector detector("yolov8n.onnx", "npu");

    // 讀取影像
    cv::Mat image = cv::imread("image.jpg");

    // 執行推理
    auto results = detector.predict(image, 0.5f);

    // 處理結果
    for (const auto& det : results.detections) {
        std::cout << det.label << ": " << det.confidence * 100 << "%" << std::endl;
        cv::rectangle(image, det.bbox.to_rect(), cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("output.jpg", image);
    return 0;
}
```

### C++ 影像分類

```cpp
#include "ivit/vision/classifier.hpp"

Classifier classifier("resnet50.onnx", "cpu");
auto results = classifier.predict(image, 5);  // Top-5

for (const auto& cls : results.classifications) {
    std::cout << cls.label << ": " << cls.score * 100 << "%" << std::endl;
}
```

### C++ 語意分割

```cpp
#include "ivit/vision/segmentor.hpp"

Segmentor segmentor("deeplabv3.onnx", "npu");
auto results = segmentor.predict(image);

cv::Mat overlay = results.overlay_mask(image, 0.5);
cv::imwrite("segmentation.jpg", overlay);
```

### Python 範例

```python
import ivit

# 物件偵測
detector = ivit.Detector("yolov8n.onnx", device="npu")
results = detector.predict("image.jpg")

for det in results.detections:
    print(f"{det.label}: {det.confidence:.2%}")

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
    Detector(const std::string& model_path,
             const std::string& device = "auto",
             const LoadConfig& config = LoadConfig{});

    Results predict(const cv::Mat& image,
                    float conf_threshold = 0.5f,
                    float iou_threshold = 0.45f);

    Results predict(const std::string& image_path,
                    float conf_threshold = 0.5f,
                    float iou_threshold = 0.45f);

    std::vector<Results> predict_batch(const std::vector<cv::Mat>& images,
                                       const InferConfig& config = InferConfig{});

    void predict_video(const std::string& source,
                       std::function<void(const Results&, const cv::Mat&)> callback,
                       const InferConfig& config = InferConfig{});

    const std::vector<std::string>& classes() const;
    int num_classes() const;
    cv::Size input_size() const;
};
```

### Results API

```cpp
class Results {
public:
    std::vector<Detection> detections;
    std::vector<ClassificationResult> classifications;
    cv::Mat segmentation_mask;

    float inference_time_ms;
    std::string device_used;
    cv::Size image_size;

    const ClassificationResult& top1() const;
    std::vector<ClassificationResult> topk(int k) const;

    std::vector<Detection> filter_by_class(const std::vector<int>& class_ids) const;
    std::vector<Detection> filter_by_confidence(float min_conf) const;

    cv::Mat colorize_mask(const std::map<int, cv::Vec3b>& colormap = {}) const;
    cv::Mat overlay_mask(const cv::Mat& image, float alpha = 0.5) const;

    cv::Mat visualize(const cv::Mat& image,
                      bool show_labels = true,
                      bool show_confidence = true,
                      bool show_boxes = true,
                      bool show_masks = true) const;

    std::string to_json() const;
    void save(const std::string& path, const std::string& format = "json") const;
};
```

### Detection 結構

```cpp
struct Detection {
    BBox bbox;
    int class_id;
    std::string label;
    float confidence;
    std::optional<cv::Mat> mask;
};

struct BBox {
    float x1, y1, x2, y2;

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

auto best = manager.get_best_device("detection", "performance");
std::cout << "Best device: " << best.id << std::endl;
```

### 裝置選擇策略

```cpp
// 明確指定裝置
Detector detector("model.onnx", "npu");

// 自動選擇
Detector detector("model.onnx", "auto");

// 指定後端
LoadConfig config;
config.backend = "openvino";
config.device = "npu";
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

### 精度設定

| 精度 | 說明 | 適用場景 |
|------|------|----------|
| `FP32` | 32 位元浮點 | 最高精度 |
| `FP16` | 16 位元浮點 | 平衡精度與效能（NPU 推薦）|
| `INT8` | 8 位元整數 | 最高效能（需校準） |

```cpp
LoadConfig config;
config.precision = "fp16";
Detector detector("model.onnx", "npu", config);
```

---

## 效能優化

### 1. 選擇正確的裝置

```
效能排序（Intel 平台）:
NPU (效率最佳) > iGPU > CPU
```

### 2. 使用 FP16 精度

```cpp
LoadConfig config;
config.precision = "fp16";
Detector detector("model.onnx", "npu", config);
```

### 3. 模型預熱

```cpp
// 預熱 3 次，避免首次推理延遲（NPU 尤其重要）
detector.model()->warmup(3);
```

### 4. 啟用模型快取

```cpp
LoadConfig config;
config.use_cache = true;
config.cache_dir = "./cache";
Detector detector("model.onnx", "npu", config);
```

### 5. 效能測試

```bash
./bin/simple_inference benchmark yolov8n.onnx npu 100
```

---

## 範例程式

### C++ 範例

| 範例 | 說明 |
|------|------|
| `simple_inference` | 綜合範例（分類、偵測、分割、效能測試） |
| `classification_demo` | 影像分類範例 |
| `detection_demo` | 物件偵測範例 |
| `video_demo` | 即時影片偵測範例 |

```bash
# 列出可用裝置
./bin/simple_inference devices

# 物件偵測（使用 NPU）
./bin/simple_inference detect yolov8n.onnx image.jpg npu output.jpg

# 影像分類
./bin/simple_inference classify resnet50.onnx cat.jpg cpu

# 語意分割
./bin/simple_inference segment deeplabv3.onnx scene.jpg npu segmented.jpg

# 效能測試
./bin/simple_inference benchmark yolov8n.onnx npu 100
```

### Python 範例

| 範例 | 說明 |
|------|------|
| `01_quickstart.py` | 快速入門範例 |
| `02_detection.py` | 物件偵測完整範例 |
| `02_classification.py` | 影像分類範例 |

```bash
# 快速入門
python examples/python/01_quickstart.py

# 物件偵測（使用 NPU）
python examples/python/02_detection.py \
    -m yolov8n.onnx \
    -i image.jpg \
    -d npu
```

---

## 常見問題

### Q: cmake 找不到 OpenVINO？

確認已安裝 APT 版本的 OpenVINO：

```bash
# 檢查 CMake 設定檔是否存在
ls /usr/lib/x86_64-linux-gnu/cmake/openvino/

# 若不存在，安裝 dev 套件
sudo apt install libopenvino-dev-2025.4.1
```

> **注意**：`pip install openvino` 安裝的是 Python 版本，會導致 C++ ABI 不相容。
> C++ 編譯必須使用 APT 安裝的版本。

### Q: 如何確認 NPU 是否可用？

```bash
# 檢查 NPU 裝置節點
ls /dev/accel*

# 檢查驅動
lsmod | grep intel_vpu

# 用 SDK 確認
./bin/simple_inference devices
```

### Q: 模型載入失敗怎麼辦？

1. 確認模型檔案存在且格式正確
2. 確認對應的後端已安裝
3. 檢查錯誤訊息

```cpp
try {
    Detector detector("model.onnx", "npu");
} catch (const ModelLoadError& e) {
    std::cerr << "Model load failed: " << e.what() << std::endl;
} catch (const DeviceNotFoundError& e) {
    std::cerr << "Device not found: " << e.what() << std::endl;
}
```

### Q: 如何支援自訂類別？

在模型目錄下建立 `labels.txt`：

```
person
car
bicycle
```

模型會自動讀取同目錄下的 `labels.txt`。

---

## 聯絡資訊

- **GitHub**: https://github.com/innodisk-mannywang/ivit-sdk
- **Issues**: https://github.com/innodisk-mannywang/ivit-sdk/issues
- **Email**: support@innodisk.com

---

*iVIT-SDK v1.0.0 - Innodisk Corporation*
