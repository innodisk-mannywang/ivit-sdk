# iVIT-SDK

**Innodisk Vision Intelligence Toolkit** - 統一的電腦視覺推論與訓練 SDK

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org)

## 概述

iVIT-SDK 是宜鼎科技開發的統一電腦視覺 SDK，提供跨硬體平台的一致性 API 介面。無論您使用的是 Intel、NVIDIA 還是 Qualcomm 的硬體，都可以使用相同的程式碼進行開發。

### 特色

- **統一 API** - 一套程式碼，支援多種硬體平台
- **多後端支援** - Intel OpenVINO、NVIDIA TensorRT、Qualcomm SNPE
- **完整視覺任務** - 分類、物件偵測、語意分割、姿態估計
- **遷移式學習** - 支援模型微調和訓練
- **雙語言支援** - Python 和 C++ API
- **自動裝置選擇** - 智慧型硬體偵測與最佳化

## 硬體支援

| 廠商 | 硬體類型 | 後端 | 架構 |
|------|---------|------|------|
| Intel | CPU、iGPU、NPU、VPU | OpenVINO | x86_64、ARM64 |
| NVIDIA | GPU | TensorRT | x86_64、ARM64 |
| Qualcomm | NPU、DSP、GPU | SNPE | ARM64 |

## 安裝

### 使用 pip 安裝

```bash
pip install ivit-sdk

# (選用) 安裝 Model Zoo 支援（自動下載和轉換模型）
pip install ultralytics
```

### 從原始碼建置

```bash
# Clone 專案
git clone https://github.com/innodisk-ai/ivit-sdk.git
cd ivit-sdk

# 建立建置目錄
mkdir build && cd build

# 設定 CMake（根據需要啟用後端）
cmake .. \
    -DIVIT_USE_OPENVINO=ON \
    -DIVIT_USE_TENSORRT=ON \
    -DIVIT_USE_ONNXRUNTIME=ON \
    -DIVIT_BUILD_PYTHON=ON

# 建置
make -j$(nproc)

# 安裝
sudo make install
```

### Python 開發模式安裝

```bash
pip install -e .

# (選用) 安裝 Model Zoo 支援（自動下載和轉換模型）
pip install -e ".[zoo]"
# 或直接安裝: pip install ultralytics
```

## 快速開始

### Python 範例

```python
import ivit

# 列出可用裝置
print("Available devices:")
ivit.devices()  # 顯示裝置列表

# 自動選擇最佳裝置
best = ivit.devices.best()
print(f"Best device: {best.name} ({best.backend})")

# 載入模型（Ultralytics 風格）
model = ivit.load("yolov8n.onnx", device=best)

# 執行推論
results = model("image.jpg")

# 處理結果
print(f"Found {len(results)} objects")
for det in results:
    print(f"{det.label}: {det.confidence:.2%}")
    print(f"  BBox: ({det.bbox.x1}, {det.bbox.y1}) - ({det.bbox.x2}, {det.bbox.y2})")

# 視覺化結果
results.show()
results.save("output.jpg")
```

### C++ 範例

```cpp
#include <ivit/ivit.hpp>
#include <opencv2/opencv.hpp>

int main() {
    // 列出可用裝置
    auto devices = ivit::list_devices();
    for (const auto& dev : devices) {
        std::cout << dev.name << " (" << dev.backend << ")" << std::endl;
    }

    // 自動選擇最佳裝置
    auto best_device = ivit::get_best_device();

    // 載入模型
    ivit::LoadConfig config;
    config.device = best_device.id;
    auto model = ivit::load_model("yolov8n.onnx", config);

    // 載入影像並推論
    cv::Mat image = cv::imread("image.jpg");
    auto results = model->predict(image);

    // 處理結果
    for (const auto& det : results.detections) {
        std::cout << det.label << ": " << det.confidence << std::endl;
    }

    // 儲存結果
    cv::imwrite("output.jpg", image);

    return 0;
}
```

## 支援的模型

### 分類 (Classification)
- EfficientNet (B0-B7)
- MobileNet V2/V3
- ResNet (18/34/50/101/152)
- VGG (16/19)

### 物件偵測 (Object Detection)
- YOLOv5 (n/s/m/l/x)
- YOLOv8 (n/s/m/l/x)
- SSD MobileNet
- Faster R-CNN

### 語意分割 (Semantic Segmentation)
- DeepLab V3+
- U-Net
- SegFormer

### 姿態估計 (Pose Estimation)
- YOLOv8-Pose
- HRNet
- OpenPose

### 臉部分析 (Face Analysis)
- RetinaFace (偵測)
- ArcFace (辨識)
- Face Landmark

## 專案結構

```
ivit-sdk/
├── include/ivit/          # C++ 標頭檔
│   ├── core/              # 核心元件
│   ├── vision/            # 視覺任務
│   ├── runtime/           # 推論執行時期
│   └── utils/             # 工具函式
├── src/                   # C++ 原始碼
├── python/ivit/           # Python 套件
│   ├── core/              # 核心模組
│   ├── vision/            # 視覺模組
│   ├── runtime/           # 執行時期模組
│   └── utils/             # 工具模組
├── tests/                 # 測試程式碼
├── examples/              # 範例程式
├── docs/                  # 文件
│   ├── PRD/               # 產品需求文件
│   ├── architecture/      # 架構文件
│   └── api/               # API 規格
└── models/                # 預訓練模型
```

## 範例程式

### 基本範例

編譯後可使用以下範例程式：

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

# 即時影片偵測
./video_demo yolov8n.onnx 0 cuda:0  # 使用攝影機 0
```

### 依角色的開發範例

我們為不同角色的開發者提供專屬範例，涵蓋從快速整合到效能優化的完整場景。

#### Python 範例 (`examples/python/`)

| 範例 | 對象 | 說明 |
|------|------|------|
| `si_quickstart.py` | 系統整合商 | 裝置探索、錯誤處理、JSON 輸出 |
| `ai_developer_training.py` | AI 應用開發者 | 遷移式學習、模型訓練、匯出 |
| `embedded_optimization.py` | 嵌入式工程師 | Runtime 配置、效能測試、自訂前處理器 |
| `backend_service.py` | 後端工程師 | Callback 監控、REST API 服務 |
| `data_analysis.py` | 資料科學家 | 結果分析、批次處理、Model Zoo |

```bash
# 系統整合商：快速整合
python examples/python/si_quickstart.py --image test.jpg

# AI 開發者：訓練自訂模型
python examples/python/ai_developer_training.py --dataset ./my_dataset --epochs 20

# 嵌入式工程師：效能優化
python examples/python/embedded_optimization.py --benchmark --iterations 100

# 後端工程師：啟動 REST API 服務
python examples/python/backend_service.py --serve --port 8080

# 資料科學家：結果分析
python examples/python/data_analysis.py --task detect --image test.jpg
```

#### C++ 範例 (`examples/cpp/`)

| 範例 | 對象 | 說明 |
|------|------|------|
| `si_quickstart.cpp` | 系統整合商 | 裝置探索、錯誤處理、JSON 序列化 |
| `embedded_optimization.cpp` | 嵌入式工程師 | Runtime 配置、Benchmark、自訂前處理器 |
| `backend_service.cpp` | 後端工程師 | Callback 系統、FPS/延遲監控 |
| `data_analysis.cpp` | 資料科學家 | 結果分析、批次處理、統計 |

```bash
# 建置範例
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 系統整合商：快速整合
./si_quickstart image.jpg model.onnx

# 嵌入式工程師：效能優化
./embedded_optimization model.onnx --device cuda:0 --benchmark --iterations 100

# 後端工程師：服務監控示範
./backend_service model.onnx --device cuda:0 --demo

# 資料科學家：結果分析
./data_analysis model.onnx image.jpg --batch
```

> **Note**: C++ 範例不包含訓練功能，因為訓練主要依賴 PyTorch 生態系統 (Python)。C++ 專注於推論與部署優化。

## 文件

- [快速入門指南](docs/GETTING_STARTED.md)
- [產品需求文件 (PRD)](docs/PRD/PRD-001-ivit-sdk.md)
- [系統架構設計](docs/architecture/ADR-001-system-architecture.md)
- [API 規格](docs/api/API-SPEC-001-ivit-sdk.md)

## 開發

### 執行測試

```bash
# Python 測試
pytest tests/python -v

# C++ 測試
cd build
ctest -V
```

### 程式碼風格

```bash
# Python
black python/
isort python/

# C++
clang-format -i src/**/*.cpp include/**/*.hpp
```

## 授權

本專案採用 [Apache License 2.0](LICENSE) 授權。

## 關於宜鼎

[宜鼎科技](https://www.innodisk.com) 是全球領先的工業級儲存和嵌入式周邊解決方案供應商，致力於提供高品質的 AI 運算平台和解決方案。

---

Copyright (c) 2024 Innodisk Corporation. All rights reserved.
