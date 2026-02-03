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
    libopenvino-intel-gpu-plugin-2025.4.1 \
    libopenvino-intel-npu-plugin-2025.4.1 \
    libopenvino-onnx-frontend-2025.4.1 \
    intel-opencl-icd
```

> **iGPU 支援**：`libopenvino-intel-gpu-plugin` 提供 Intel iGPU 推論支援，`intel-opencl-icd` 提供 Intel OpenCL runtime。
> 若系統同時有 NVIDIA dGPU 與 Intel iGPU，需安裝這兩個套件才能讓 OpenVINO 正確偵測到 Intel iGPU。
>
> **NPU 支援**：`libopenvino-intel-npu-plugin` 提供 Intel NPU 推論支援。
> 若不需要 NPU，可以省略該套件。

### Step 3：（選用）安裝 TensorRT（NVIDIA GPU）

#### x86 + NVIDIA dGPU（Ubuntu 22.04/24.04）

```bash
# 加入 NVIDIA CUDA APT 來源
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
# Ubuntu 24.04 請將 ubuntu2204 替換為 ubuntu2404

sudo apt update

# 安裝 CUDA Toolkit + TensorRT 開發套件
sudo apt install cuda-toolkit tensorrt-dev
```

> 若只需要最小安裝（C++ 編譯用）：
> ```bash
> sudo apt install libnvinfer-dev libnvonnxparsers-dev libnvinfer-plugin-dev
> ```

#### NVIDIA Jetson（JetPack）

Jetson 裝置透過 JetPack SDK 已預裝 CUDA、cuDNN、TensorRT，不需額外安裝：

```bash
# 確認 JetPack 已安裝或更新到最新版
sudo apt update
sudo apt install nvidia-jetpack

# 驗證 TensorRT
dpkg -l | grep nvinfer
```

> **JetPack 6.2** 預裝 TensorRT 10.3 + CUDA 12.6。

### Step 4：（選用）安裝 Python 綁定依賴

若需要編譯 Python binding（`-DIVIT_BUILD_PYTHON=ON`）：

```bash
# Ubuntu 22.04
sudo apt install python3-dev
pip install pybind11

# Ubuntu 24.04（系統 Python 不允許 pip install，改用 apt）
sudo apt install python3-dev python3-pybind11
```

### 驗證安裝

```bash
# 確認 cmake 版本
cmake --version

# 確認 OpenCV
pkg-config --modversion opencv4

# 確認 OpenVINO CMake 可被找到
ls /usr/lib/cmake/openvino2025.4.1/

# （選用）確認 CUDA + TensorRT
nvcc --version
dpkg -l | grep nvinfer
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

### 安裝 Python 套件

編譯完成後，安裝 Python 套件以使用 `ivit` CLI 和 Python API：

```bash
# Ubuntu 22.04
pip install -e ".[zoo]"

# Ubuntu 24.04（需要使用 venv）
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[zoo]"
```

> **注意**：Ubuntu 24.04 的系統 Python 不允許直接 `pip install`（PEP 668），必須使用 venv。
> 之後每次開新 terminal 需先執行 `source .venv/bin/activate`。

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
| `FP32` | 32 位元浮點 | 最高精度，NPU/CPU 推薦直接使用 |
| `FP16` | 16 位元浮點 | 縮小模型檔案（約 50%），適合儲存與部署 |
| `INT8` | 8 位元整數 | 最高效能（需校準資料集） |

```cpp
LoadConfig config;
config.precision = "fp16";
Detector detector("model.onnx", "npu", config);
```

### 模型轉換（`ivit convert`）

SDK 提供 CLI 工具將 ONNX 模型轉換為各後端的優化格式。轉換優先使用 C++ binding（與 APT 安裝的 OpenVINO 相容），若 C++ binding 不可用則退回命令列工具（`ovc`）。

#### 基本用法

```bash
# ONNX → OpenVINO IR（FP32）
ivit convert model.onnx --format openvino

# ONNX → OpenVINO IR（FP16 壓縮）
ivit convert model.onnx --format openvino --precision fp16

# ONNX → TensorRT Engine
ivit convert model.onnx --format tensorrt --precision fp16

# 指定輸出目錄
ivit convert model.onnx --format openvino -o ./output/
```

#### 完整轉換流程（含編譯）

首次使用或更新 SDK 後，需先編譯再轉換：

```bash
# 1. 編譯 C++ 核心與 Python binding
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DIVIT_BUILD_PYTHON=ON
make -j$(nproc)
cd ..

# 2. 安裝系統 library
sudo cp build/lib/libivit.so /usr/local/lib/
sudo ldconfig

# 3. 安裝 Python 套件
pip install -e .

# 4. 執行轉換
ivit convert model.onnx --format openvino --precision fp16
```

#### 精度支援與限制

| 精度 | OpenVINO | TensorRT | 說明 |
|------|----------|----------|------|
| `fp32` | ✅ | ✅ | 預設精度 |
| `fp16` | ✅ | ✅ | 縮小模型檔案，不影響推論速度（見下方說明） |
| `int8` | ❌ | ✅（需校準） | OpenVINO INT8 需使用 [NNCF](https://github.com/openvinotoolkit/nncf) 工具 |

> **INT8 限制**：OpenVINO 的 INT8 量化需要校準資料集，無法透過 `ivit convert` 直接轉換。
> 請使用 [NNCF toolkit](https://github.com/openvinotoolkit/nncf) 進行 INT8 量化。

#### FP16 轉換效能說明

FP16 壓縮**不會提升推論速度**，唯一好處是模型檔案大小減半。原因如下：

- **NPU**：硬體原生以 FP16 計算，不論輸入模型是 FP32 或 FP16，NPU 編譯器會自動處理精度轉換。使用 FP16 IR 反而因額外的 Convert 節點可能略慢。
- **CPU**：x86 CPU 原生以 FP32 計算，FP16 權重會在推論時解壓回 FP32，無效能增益。

以下為 YOLOv8n 在 Intel AIPC 平台的實測數據（100 次迭代）：

| 模型格式 | 裝置 | 平均延遲 | 吞吐量 | 檔案大小 |
|----------|------|----------|--------|----------|
| ONNX (FP32) | NPU | 8.74 ms | 114 FPS | 13 MB |
| IR FP32 | NPU | 8.77 ms | 114 FPS | ~12 MB |
| IR FP16 | NPU | 13.38 ms | 75 FPS | ~6 MB |
| ONNX (FP32) | CPU | 33.25 ms | 30 FPS | 13 MB |
| IR FP32 | CPU | 34.30 ms | 29 FPS | ~12 MB |
| IR FP16 | CPU | 34.18 ms | 29 FPS | ~6 MB |

> **建議**：NPU 推論直接使用 ONNX 或 FP32 IR 即可，效能最佳。
> FP16 轉換適合需要縮小模型檔案的部署場景（如嵌入式裝置儲存空間有限）。

#### 轉換機制

`ivit convert` 的轉換策略依優先順序：

1. **C++ binding**（推薦）：使用 SDK 內建的 C++ `convert_model()` 函式，直接呼叫 OpenVINO/TensorRT C++ API。適用於 APT 安裝的 OpenVINO 環境。
2. **命令列工具**（備援）：OpenVINO 使用 `ovc` 工具，TensorRT 使用 Python `tensorrt` 模組。

#### C++ / Python API 轉換

除 CLI 外，也可在程式中直接呼叫轉換 API：

```cpp
// C++
#include "ivit/ivit.hpp"

ivit::convert_model("yolov8n.onnx", "yolov8n.xml", "cpu", "fp16");
```

```python
# Python
import ivit
ivit.convert_model("yolov8n.onnx", "yolov8n.xml", "cpu", "fp16")
```

---

## 效能優化

### 1. 選擇正確的裝置

```
效能排序（Intel 平台）:
NPU (效率最佳) > iGPU > CPU
```

### 2. 直接使用 ONNX 或 FP32 IR

NPU 硬體原生以 FP16 計算，不需要事先轉換模型精度。直接使用 ONNX 即可獲得最佳效能：

```cpp
// 直接載入 ONNX，NPU 會自動以 FP16 計算
Detector detector("model.onnx", "npu");
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
| `02_segmentation.py` | 語意分割範例 |

Python 範例需要先下載模型。有兩種方式：

**方式 1：使用環境變數指定模型路徑（推薦）**

```bash
# 下載模型到 zoo 快取目錄
ivit zoo download yolov8n

# 透過環境變數指定模型路徑
IVIT_MODEL_PATH=~/.cache/ivit/models/yolov8n.onnx python examples/python/01_quickstart.py
```

**方式 2：將模型放到範例預設路徑**

```bash
mkdir -p models/onnx
ivit zoo download yolov8n
cp ~/.cache/ivit/models/yolov8n.onnx models/onnx/
python examples/python/01_quickstart.py
```

```bash
# 物件偵測（使用 NPU）
ivit zoo download yolov8n
python examples/python/02_detection.py \
    -m ~/.cache/ivit/models/yolov8n.onnx \
    -i examples/data/images/bus.jpg \
    -d npu

# 影像分類（使用 NPU）
ivit zoo download yolov8n-cls
python examples/python/02_classification.py \
    -m ~/.cache/ivit/models/yolov8n-cls.onnx \
    -i examples/data/images/bus.jpg \
    -d npu

# 語意分割（使用 NPU）
ivit zoo download deeplabv3-mobilenetv3
python examples/python/02_segmentation.py \
    -m ~/.cache/ivit/models/deeplabv3-mobilenetv3.onnx \
    -i examples/data/images/bus.jpg \
    -d npu
```

---

## 常見問題

### Q: cmake 找不到 OpenVINO？

確認已安裝 APT 版本的 OpenVINO：

```bash
# 檢查 CMake 設定檔是否存在
ls /usr/lib/cmake/openvino2025.4.1/

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
