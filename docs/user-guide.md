# iVIT-SDK 開發者指南

> **版本**：1.0.0
> **更新日期**：2026-01-26
> **作者**：Innodisk AI Team

---

## 目錄

1. [簡介](#簡介)
2. [安裝指南](#安裝指南)
3. [快速入門](#快速入門)
4. [依角色的開發指南](#依角色的開發指南)
   - [系統整合商 (SI)](#系統整合商-si)
   - [AI 應用開發者](#ai-應用開發者)
   - [嵌入式工程師](#嵌入式工程師)
   - [後端工程師](#後端工程師)
   - [資料科學家](#資料科學家)
5. [核心 API 參考](#核心-api-參考)
6. [最佳實務](#最佳實務)
7. [故障排除](#故障排除)
8. [Model Zoo 完整清單](#model-zoo-完整清單)
9. [附錄](#附錄)

---

## 簡介

### 什麼是 iVIT-SDK？

**iVIT-SDK**（Innodisk Vision Intelligence Toolkit）是宜鼎國際為 AI 運算平台開發的統一電腦視覺推論與訓練 SDK。本 SDK 提供跨硬體平台的統一 API 介面，讓開發者能夠「**一次開發，多平台部署**」。

### 核心價值

| 特色 | 說明 |
|------|------|
| **統一 API** | 無論 Intel 或 NVIDIA，使用相同的程式碼（Qualcomm 規劃中） |
| **極簡設計** | 類似 Ultralytics 的一行載入、一行推論風格 |
| **遷移式學習** | 內建訓練模組，支援模型微調 |
| **多任務支援** | 分類、偵測、分割、姿態估計 |
| **雙語 API** | Python 和 C++ 功能對等 |

### 支援的硬體平台

| 廠商 | 硬體 | 推論引擎 | x86 | ARM |
|------|------|----------|:---:|:---:|
| Intel | CPU / iGPU / NPU / VPU | OpenVINO | ✅ | ✅ |
| NVIDIA | dGPU / Jetson | TensorRT | ✅ | ✅ |
| Qualcomm | IQ9/IQ8/IQ6 (Hexagon NPU) | QNN (規劃中) | - | ✅ |

---

## 安裝指南

### 系統需求

- **作業系統**：Ubuntu 20.04+ / Windows 10+
- **Python**：3.9 或更高版本
- **硬體**：支援的 Intel、NVIDIA 或 Qualcomm 裝置

### Python 安裝

#### 方法 1：從原始碼安裝（推薦）

```bash
# 複製專案
git clone https://github.com/innodisk-mannywang/ivit-sdk.git
cd ivit-sdk

# 基本安裝
pip install -e .

# (選用) 安裝 Model Zoo 支援（自動下載和轉換模型）
pip install -e ".[zoo]"

# 包含訓練功能（需要 PyTorch）
pip install -e ".[train]"

# 包含開發工具
pip install -e ".[dev]"

# 包含所有功能
pip install -e ".[all]"
```

#### 方法 2：使用 pip 安裝（套件發布後）

```bash
# 基本安裝
pip install ivit-sdk

# (選用) 安裝 Model Zoo 支援
pip install "ivit-sdk[zoo]"

# 包含訓練功能
pip install "ivit-sdk[train]"

# 包含所有功能
pip install "ivit-sdk[all]"

# 安裝特定後端支援
pip install "ivit-sdk[openvino]"    # Intel OpenVINO
pip install "ivit-sdk[tensorrt]"    # NVIDIA TensorRT
```

> **注意**：目前套件尚未發布至 PyPI，請使用方法 1 從原始碼安裝。

> **Model Zoo 說明**：使用 `ivit.zoo.load()` 自動下載模型時，需要安裝 `ultralytics` 套件來進行模型轉換。若使用本地 ONNX 檔案則不需要。

### C++ 建置

#### 系統需求

- **編譯器**：GCC 9+ / Clang 10+ / MSVC 2019+
- **CMake**：3.16 或更高版本
- **OpenCV**：4.5 或更高版本

#### 依賴套件安裝 (Ubuntu)

```bash
# 基本建置工具
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# OpenCV
sudo apt-get install -y libopencv-dev

# (選用) Intel OpenVINO
# 參考：https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html

# (選用) NVIDIA TensorRT
# 參考：https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
```

#### 建置步驟

```bash
# 複製專案
git clone https://github.com/innodisk-mannywang/ivit-sdk.git
cd ivit-sdk

# 建立建置目錄
mkdir build && cd build

# 設定 CMake（根據需要啟用後端）
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_USE_OPENVINO=ON \
    -DIVIT_USE_TENSORRT=ON \
    -DIVIT_BUILD_EXAMPLES=ON

# 建置
make -j$(nproc)

# 安裝（選用）
sudo make install
```

#### CMake 選項

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `DIVIT_USE_OPENVINO` | OFF | 啟用 Intel OpenVINO 後端 |
| `DIVIT_USE_TENSORRT` | OFF | 啟用 NVIDIA TensorRT 後端 |
| `DIVIT_USE_QNN` | OFF | 啟用 Qualcomm QNN 後端 (IQ Series) |
| `DIVIT_BUILD_EXAMPLES` | ON | 建置範例程式 |
| `DIVIT_BUILD_TESTS` | OFF | 建置測試程式 |
| `DIVIT_BUILD_PYTHON` | OFF | 建置 Python 綁定 |

#### 在專案中使用

**CMakeLists.txt**：

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_project)

# 尋找 iVIT-SDK
find_package(ivit REQUIRED)

# 建立執行檔
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE ivit::ivit)
```

**pkg-config**：

```bash
# 編譯
g++ -o my_app main.cpp $(pkg-config --cflags --libs ivit)
```

### 驗證安裝

#### Python

```python
import ivit

# 檢查版本
print(f"iVIT-SDK 版本: {ivit.__version__}")

# 檢查可用裝置
ivit.devices()
```

預期輸出：
```
iVIT-SDK 版本: 1.0.0
╭─────────────────────────────────────────────────────────╮
│                   iVIT Available Devices                │
├─────────────────────────────────────────────────────────┤
│  ID       │ Name                    │ Backend          │
├─────────────────────────────────────────────────────────┤
│  cpu      │ Intel(R) Xeon(R)        │ openvino         │
│  cuda:0   │ NVIDIA RTX 6000 Ada     │ tensorrt         │
│  cuda:1   │ NVIDIA RTX 6000 Ada     │ tensorrt         │
╰─────────────────────────────────────────────────────────╯
```

#### C++

```cpp
#include <iostream>
#include "ivit/ivit.hpp"

int main() {
    // 檢查版本
    std::cout << "iVIT-SDK Version: " << ivit::version() << std::endl;

    // 列出可用裝置
    auto devices = ivit::list_devices();
    std::cout << "Available devices: " << devices.size() << std::endl;

    for (const auto& dev : devices) {
        std::cout << "  - " << dev.id << ": " << dev.name
                  << " (" << dev.backend << ")" << std::endl;
    }

    return 0;
}
```

建置與執行：
```bash
cd build
./simple_inference devices
```

預期輸出：
```
iVIT-SDK Version: 1.0.0
Available devices: 3
  - cpu: Intel(R) Xeon(R) (openvino)
  - cuda:0: NVIDIA RTX 6000 Ada (tensorrt)
  - cuda:1: NVIDIA RTX 6000 Ada (tensorrt)
```

---

## 快速入門

### 30 秒快速體驗

**方式一：使用 Model Zoo（推薦新手）**

```python
import ivit

# 從 Model Zoo 載入模型（自動下載）
model = ivit.zoo.load("yolov8n")

# 執行推論
results = model("image.jpg")

# 顯示結果
results.show()
```

> **可用模型**：`yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`, `yolov8n-cls`, `yolov8s-cls`, `resnet50`, `mobilenetv3`, `efficientnet-b0`, `yolov8n-seg`, `yolov8s-seg`, `yolov8n-pose`, `yolov8s-pose`
>
> 完整清單與效能指標請參考 [Model Zoo 完整清單](#model-zoo-完整清單)。

**方式二：使用自己的模型檔案**

```python
import ivit

# 載入本地模型檔案（需自備 .onnx/.xml/.engine 檔案）
model = ivit.load("path/to/your/model.onnx")

# 執行推論
results = model("image.jpg")

# 顯示結果
results.show()
```

> **⚠️ 自定義模型注意事項**
>
> 使用非 Model Zoo 的模型時，可能遇到以下問題：
>
> | 問題類型 | 說明 | 解決方案 |
> |----------|------|----------|
> | **不支援的運算子** | 模型包含推論引擎不支援的 Op | 確認 ONNX opset 版本，或簡化模型結構 |
> | **前處理不匹配** | 輸入格式與預期不符（RGB/BGR、正規化方式） | 使用 `model.set_preprocessor()` 自定義前處理 |
> | **後處理不匹配** | 輸出格式與內建解析器不符 | 使用 `model.set_postprocessor()` 自定義後處理 |
> | **輸入形狀錯誤** | 模型預期固定尺寸或動態尺寸 | 檢查模型輸入規格，調整圖像尺寸 |
> | **自定義層** | 模型包含自定義實作的層 | 需將自定義層轉為標準運算子組合 |
>
> **建議做法**：
> ```python
> # 1. 先檢查模型資訊
> model = ivit.load("custom_model.onnx")
> print(f"輸入: {model.input_info}")
> print(f"輸出: {model.output_info}")
>
> # 2. 如果前後處理不匹配，自定義處理器
> from ivit.core.processors import BasePreProcessor, BasePostProcessor
>
> class MyPreProcessor(BasePreProcessor):
>     def __call__(self, image):
>         # 實作您的前處理邏輯
>         ...
>
> class MyPostProcessor(BasePostProcessor):
>     def __call__(self, outputs, original_shape):
>         # 實作您的後處理邏輯
>         ...
>
> model.set_preprocessor(MyPreProcessor())
> model.set_postprocessor(MyPostProcessor())
> ```
>
> 詳細說明請參考 [嵌入式工程師 - 自定義前處理器](#自定義前處理器) 章節。
>
> **📘 完整教學**：[自定義模型整合指南](./tutorials/custom-model.md) - 包含三個實際範例：前後處理不匹配、不支援的運算子、自定義輸出格式。

> **Model Zoo 說明**：iVIT 內建 14 個預訓練模型，使用 `ivit.zoo.load()` 會自動下載模型到快取目錄並自動配置正確的前後處理。**推薦新手優先使用 Model Zoo**。詳見 [Model Zoo 完整清單](#model-zoo-完整清單)。

### 完整範例

```python
import ivit

# 1. 探索可用裝置
print("可用裝置:")
ivit.devices()

# 2. 自動選擇最佳裝置
best_device = ivit.devices.best()
print(f"最佳裝置: {best_device.id} ({best_device.backend})")

# 3. 查看 Model Zoo 可用模型
print("可用模型:")
print(ivit.zoo.list_models())

# 4. 從 Model Zoo 載入模型（自動下載 + 指定裝置）
model = ivit.zoo.load("yolov8n", device=best_device)

# 5. 執行推論
results = model("image.jpg")

# 6. 處理結果
print(f"偵測到 {len(results)} 個物件")
for det in results:
    print(f"  - {det.label}: {det.confidence:.2%}")

# 7. 視覺化
results.show()

# 8. 儲存結果
results.save("output.jpg")
```

---

## 依角色的開發指南

不同角色的開發者有不同的需求。以下針對五種常見角色提供專屬的開發指南。

---

### 系統整合商 (SI)

> **目標**：快速整合 AI 推論功能到現有系統中

#### 使用情境

- 需要在短時間內完成 POC
- 客戶環境多樣，需要跨平台支援
- 重視 API 的穩定性和錯誤處理

#### 快速整合範例

```python
import ivit

# ============================================================
# 情境：5 分鐘內完成基本推論整合
# ============================================================

# 步驟 1：檢查可用裝置
devices = ivit.devices()
print(f"找到 {len(devices)} 個可用裝置")

# 步驟 2：自動選擇最佳裝置（無需手動指定）
best = ivit.devices.best()
print(f"自動選擇: {best.id} ({best.name})")

# 步驟 3：載入模型
# 方式 A：使用 Model Zoo（快速驗證）
model = ivit.zoo.load("yolov8n", device=best)

# 方式 B：使用客戶提供的模型檔案
# model = ivit.load("customer_model.onnx", device=best)

# 步驟 4：執行推論
results = model("input.jpg")

# 步驟 5：取得結構化結果
output = results.to_dict()
print(f"偵測結果: {output}")
```

#### 裝置探索 API

```python
import ivit

# 列出所有裝置
all_devices = ivit.devices()

# 取得特定類型裝置
cpu = ivit.devices.cpu()        # CPU 裝置
cuda = ivit.devices.cuda()      # NVIDIA GPU
npu = ivit.devices.npu()        # Intel NPU

# 自動選擇最佳裝置
best_perf = ivit.devices.best()                    # 效能優先
best_eff = ivit.devices.best("efficiency")         # 效率優先

# 取得裝置詳細資訊
device = ivit.devices.best()
print(f"裝置 ID: {device.id}")
print(f"裝置名稱: {device.name}")
print(f"後端引擎: {device.backend}")
print(f"裝置類型: {device.type}")
```

#### 錯誤處理機制

```python
import ivit
from ivit import (
    IVITError,
    ModelLoadError,
    DeviceNotFoundError,
    InferenceError,
    ConfigurationError,
)

def safe_inference(model_path, image_path):
    """安全的推論函數，包含完整錯誤處理"""
    try:
        # 載入模型
        model = ivit.load(model_path)

        # 執行推論
        results = model(image_path)

        return {"success": True, "results": results.to_dict()}

    except ModelLoadError as e:
        # 模型載入失敗
        return {
            "success": False,
            "error_type": "ModelLoadError",
            "message": str(e),
            "suggestion": "請確認模型路徑和格式是否正確"
        }

    except DeviceNotFoundError as e:
        # 裝置不可用
        return {
            "success": False,
            "error_type": "DeviceNotFoundError",
            "message": str(e),
            "suggestion": "請執行 ivit.devices() 確認可用裝置"
        }

    except InferenceError as e:
        # 推論錯誤
        return {
            "success": False,
            "error_type": "InferenceError",
            "message": str(e),
            "suggestion": "請檢查輸入圖像格式和尺寸"
        }

    except IVITError as e:
        # 其他 iVIT 錯誤
        return {
            "success": False,
            "error_type": "IVITError",
            "message": str(e)
        }

# 使用範例
result = safe_inference("model.onnx", "image.jpg")
if result["success"]:
    print("推論成功！")
else:
    print(f"錯誤: {result['message']}")
    print(f"建議: {result.get('suggestion', '無')}")
```

#### SI 最佳實務

1. **使用 `ivit.devices.best()` 自動選擇裝置**
2. **總是包裝錯誤處理邏輯**
3. **使用 `results.to_dict()` 取得結構化輸出**
4. **測試多種硬體環境**

#### C++ 範例

```cpp
#include "ivit/ivit.hpp"
#include <opencv2/opencv.hpp>

using namespace ivit;

int main() {
    // Step 1: 裝置探索
    auto devices = list_devices();
    std::cout << "Found " << devices.size() << " devices" << std::endl;

    auto best = get_best_device();
    std::cout << "Best device: " << best.id << std::endl;

    // Step 2: 載入模型（使用 load_model API）
    LoadConfig config;
    config.device = best.id;
    auto model = load_model("yolov8n.onnx", config);

    // Step 3: 安全推論
    try {
        cv::Mat image = cv::imread("image.jpg");
        auto results = model->predict(image);

        std::cout << "Detections: " << results.detections.size() << std::endl;
        std::cout << "Inference time: " << results.inference_time_ms << " ms" << std::endl;

        // 輸出偵測結果
        for (const auto& det : results.detections) {
            std::cout << det.label << ": " << det.confidence << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

**完整範例**：`examples/cpp/si_quickstart.cpp`

```bash
# 建置與執行
cd build && make si_quickstart
./si_quickstart image.jpg model.onnx
```

---

### AI 應用開發者

> **目標**：訓練和部署自定義 AI 模型

#### 使用情境

- 需要微調預訓練模型以適應特定場景
- 處理客戶自有資料集
- 需要完整的訓練、驗證、匯出流程

#### 完整訓練流程

```python
import ivit
from ivit.train import (
    Trainer,
    ImageFolderDataset,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
)

# ============================================================
# 情境：使用遷移式學習訓練自定義分類器
# ============================================================

# 步驟 1：準備資料集
# 資料夾結構:
# my_dataset/
#   ├── cat/
#   │   ├── image1.jpg
#   │   └── image2.jpg
#   └── dog/
#       ├── image1.jpg
#       └── image2.jpg

train_dataset = ImageFolderDataset(
    root="./my_dataset",
    train_split=0.8,
    split="train"
)
val_dataset = ImageFolderDataset(
    root="./my_dataset",
    train_split=0.8,
    split="val"
)

print(f"訓練集大小: {len(train_dataset)}")
print(f"驗證集大小: {len(val_dataset)}")
print(f"類別數: {train_dataset.num_classes}")
print(f"類別名稱: {train_dataset.class_names}")

# 步驟 2：建立訓練器
trainer = Trainer(
    model="resnet50",           # 可選: resnet18/34/50/101, efficientnet_b0-b2, mobilenet_v2/v3
    dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=20,
    learning_rate=0.001,
    batch_size=32,
    device="cuda:0",
    freeze_backbone=True,       # 遷移式學習: 凍結骨幹網路
    optimizer="adam",           # 可選: adam, adamw, sgd
)

# 步驟 3：設定回調
callbacks = [
    EarlyStopping(patience=5, monitor="val_loss"),
    ModelCheckpoint("best_model.pt", monitor="val_accuracy"),
    ProgressLogger(),
]

# 步驟 4：開始訓練
history = trainer.fit(callbacks=callbacks)

# 步驟 5：評估模型
metrics = trainer.evaluate()
print(f"最終準確率: {metrics['accuracy']:.2%}")

# 步驟 6：匯出模型
trainer.export("my_model.onnx", format="onnx", quantize="fp16")
```

#### 支援的預訓練模型

| 類別 | 模型 |
|------|------|
| ResNet | resnet18, resnet34, resnet50, resnet101 |
| EfficientNet | efficientnet_b0, efficientnet_b1, efficientnet_b2 |
| MobileNet | mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large |
| VGG | vgg16, vgg19 |
| DenseNet | densenet121 |

#### 資料增強

```python
from ivit.train import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ColorJitter,
    Normalize,
    ToTensor,
)

# 自定義訓練增強
train_augmentation = Compose([
    Resize(224),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.2, contrast=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensor(),
])

# 驗證增強（不含隨機變換）
val_augmentation = Compose([
    Resize(224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensor(),
])

# 套用增強
image = train_augmentation(original_image)
```

#### 支援的資料集格式

**1. ImageFolder 格式**
```python
from ivit.train import ImageFolderDataset

dataset = ImageFolderDataset(
    root="./data",
    train_split=0.8,
    split="train"
)
```

**2. COCO 格式**
```python
from ivit.train import COCODataset

dataset = COCODataset(
    images_dir="./coco/images",
    annotations_file="./coco/annotations.json"
)
```

**3. YOLO 格式**
```python
from ivit.train import YOLODataset

dataset = YOLODataset(
    images_dir="./yolo/images",
    labels_dir="./yolo/labels",
    class_names=["cat", "dog", "bird"]
)
```

#### 訓練回調

```python
from ivit.train import (
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
    LRScheduler,
    TensorBoardLogger,
)

# 早停：當驗證損失不再改善時停止訓練
early_stop = EarlyStopping(
    patience=5,
    monitor="val_loss",
    min_delta=0.001
)

# 模型檢查點：儲存最佳模型
checkpoint = ModelCheckpoint(
    filepath="best_model.pt",
    monitor="val_accuracy",
    save_best_only=True
)

# 進度記錄
progress = ProgressLogger()

# 學習率調整
lr_scheduler = LRScheduler(
    schedule_type="step",
    step_size=10,
    gamma=0.1
)

# TensorBoard 日誌
tensorboard = TensorBoardLogger(log_dir="./logs")

# 使用所有回調
trainer.fit(callbacks=[early_stop, checkpoint, progress, lr_scheduler, tensorboard])
```

#### 模型匯出

```python
# 匯出為 ONNX（跨平台）
trainer.export("model.onnx", format="onnx", quantize="fp16")

# 匯出為 TorchScript
trainer.export("model.pt", format="torchscript")

# 匯出為 OpenVINO IR（Intel 優化）
trainer.export("model.xml", format="openvino", quantize="int8")

# 匯出為 TensorRT Engine（NVIDIA 優化）
trainer.export("model.engine", format="tensorrt", quantize="fp16")
```

#### C++ 說明

> **Note**: 訓練功能目前僅支援 Python API，因為底層依賴 PyTorch 生態系統。C++ API 專注於推論部署。
>
> 訓練完成後，可將模型匯出為 ONNX 格式，再使用 C++ API 進行部署：
>
> ```cpp
> // 載入 Python 訓練後匯出的模型
> ivit::LoadConfig config;
> config.device = "cuda:0";
> auto model = ivit::load_model("my_trained_model.onnx", config);
> auto results = model->predict(image);
> ```

---

### 嵌入式工程師

> **目標**：在邊緣裝置上實現低延遲、高效能推論

#### 使用情境

- 需要優化推論效能和記憶體使用
- 針對特定硬體進行調優
- 關注前後處理的效能

#### Runtime 配置

```python
import ivit

# ============================================================
# 情境：針對特定硬體優化推論效能
# ============================================================

# 載入模型（使用 Model Zoo 或本地檔案皆可）
model = ivit.zoo.load("yolov8n", device="cuda:0")
# 或使用本地檔案: model = ivit.load("yolov8n.onnx", device="cuda:0")

# --- OpenVINO 配置（Intel 硬體）---
model.configure_openvino(
    performance_mode="LATENCY",      # LATENCY 或 THROUGHPUT
    num_streams=1,                   # 推論串流數
    inference_precision="FP16",      # 精度
    enable_cpu_pinning=True,         # CPU 核心綁定
)

# --- TensorRT 配置（NVIDIA 硬體）---
model.configure_tensorrt(
    workspace_size=1 << 30,          # 1GB 工作空間
    fp16=True,                       # 啟用 FP16
    int8=False,                      # INT8 需要校正資料
    dla_core=-1,                     # DLA 核心（Jetson）
    builder_optimization_level=3,   # 優化等級（0-5）
    enable_sparsity=True,            # 稀疏加速
)

# --- QNN 配置（Qualcomm IQ Series 硬體）--- (規劃中，尚未提供)
model.configure_qnn(
    backend="htp",                   # cpu, gpu, htp (Hexagon Tensor Processor)
    performance_profile="HIGH_PERFORMANCE",
    htp_precision="fp16",            # fp16, int8
)

# 預熱推論（重要！）
model.warmup(iterations=10)
```

#### 前後處理器

```python
from ivit.core.processors import (
    get_preprocessor,
    get_postprocessor,
    register_preprocessor,
    register_postprocessor,
    BasePreProcessor,
    BasePostProcessor,
)
import numpy as np
import time

# ============================================================
# 情境：驗證和優化前後處理效能
# ============================================================

# 取得內建前處理器
letterbox = get_preprocessor("letterbox")
center_crop = get_preprocessor("center_crop")

# 取得內建後處理器
yolo_post = get_postprocessor("yolo")
cls_post = get_postprocessor("classification")

# 效能測試
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

def benchmark_preprocessor(preprocessor, image, iterations=100):
    """測試前處理器效能"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = preprocessor(image)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "平均耗時": f"{np.mean(times):.3f}ms",
        "最小耗時": f"{np.min(times):.3f}ms",
        "最大耗時": f"{np.max(times):.3f}ms",
        "標準差": f"{np.std(times):.3f}ms",
    }

# 測試 Letterbox
print("Letterbox 效能:")
print(benchmark_preprocessor(letterbox, test_image))

# 測試 CenterCrop
print("\nCenterCrop 效能:")
print(benchmark_preprocessor(center_crop, test_image))
```

#### 自定義前處理器

```python
from ivit.core.processors import BasePreProcessor, register_preprocessor
import numpy as np
import cv2

class CustomPreProcessor(BasePreProcessor):
    """自定義前處理器範例"""

    def __init__(self, target_size=(640, 640), normalize=True):
        self.target_size = target_size
        self.normalize = normalize

    def process(self, image: np.ndarray, target_size: tuple = None, **kwargs) -> tuple:
        """
        前處理圖像。

        Args:
            image: 輸入圖像 (BGR, HWC)
            target_size: 目標尺寸，若為 None 則使用 self.target_size

        Returns:
            Tuple of (tensor, preprocess_info)
        """
        if target_size is None:
            target_size = self.target_size

        orig_h, orig_w = image.shape[:2]

        # 1. 調整尺寸
        resized = cv2.resize(image, target_size)

        # 2. BGR 轉 RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. 正規化
        if self.normalize:
            rgb = rgb.astype(np.float32) / 255.0

        # 4. HWC 轉 NCHW
        transposed = np.transpose(rgb, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        # 回傳 tensor 和前處理資訊（供後處理使用）
        preprocess_info = {
            "orig_size": (orig_h, orig_w),
            "target_size": target_size,
        }

        return batched, preprocess_info

# 註冊自定義前處理器
register_preprocessor("custom", CustomPreProcessor)

# 使用自定義前處理器
model = ivit.load("model.onnx")
model.set_preprocessor(CustomPreProcessor(target_size=(416, 416)))
```

#### 自定義後處理器

```python
from ivit.core.processors import BasePostProcessor, register_postprocessor
from ivit.core.result import Results
from ivit.core.types import Detection, BBox
import numpy as np

class CustomPostProcessor(BasePostProcessor):
    """自定義後處理器範例：YOLO 輸出解析"""

    def __init__(self, conf_threshold=0.5, iou_threshold=0.45, class_names=None):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []

    def process(
        self,
        outputs: dict,
        orig_size: tuple,
        preprocess_info: dict = None,
        config=None,
        labels: list = None,
    ) -> Results:
        """
        後處理模型輸出。

        Args:
            outputs: 原始模型輸出
            orig_size: 原始圖像尺寸 (height, width)
            preprocess_info: 前處理資訊
            config: 推論配置
            labels: 類別標籤（若為 None 則使用 self.class_names）

        Returns:
            Results 物件
        """
        results = Results()
        results.image_size = orig_size

        if labels is None:
            labels = self.class_names

        # 解析模型輸出（範例）
        predictions = outputs.get("output", outputs[list(outputs.keys())[0]])

        # 過濾低信心度預測
        for pred in predictions:
            confidence = float(pred[4])
            if confidence < self.conf_threshold:
                continue

            class_id = int(pred[5])
            label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"

            det = Detection(
                bbox=BBox(pred[0], pred[1], pred[2], pred[3]),
                class_id=class_id,
                label=label,
                confidence=confidence
            )
            results.detections.append(det)

        # NMS（非極大值抑制）
        results.detections = self._nms(results.detections, self.iou_threshold)

        return results

    def _nms(self, detections, iou_threshold):
        """簡單的 NMS 實作"""
        if not detections:
            return detections

        # 按信心度排序
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if d.class_id != best.class_id or best.bbox.iou(d.bbox) < iou_threshold
            ]

        return keep

# 註冊並使用
register_postprocessor("custom_yolo", CustomPostProcessor)
model.set_postprocessor(CustomPostProcessor(conf_threshold=0.6, class_names=["person", "car"]))
```

#### 嵌入式最佳實務

1. **一定要執行 warmup** - 前幾次推論通常較慢
2. **使用 FP16 量化** - 大多數情況下精度損失可忽略
3. **根據硬體調整配置** - OpenVINO 用 LATENCY 模式，TensorRT 啟用 CUDA Graph
4. **監控前處理耗時** - 前處理可能佔總耗時 30% 以上

#### C++ 範例

```cpp
#include "ivit/ivit.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <numeric>

using namespace ivit;

int main() {
    // 選擇裝置
    auto device = get_best_device();
    std::cout << "Using device: " << device.id << std::endl;

    // 載入模型（使用 load_model API）
    LoadConfig config;
    config.device = device.id;
    auto model = load_model("yolov8n.onnx", config);

    // Step 1: 模型預熱（重要！）
    std::cout << "Warming up..." << std::endl;
    cv::Mat dummy(480, 640, CV_8UC3);
    cv::randu(dummy, cv::Scalar(0), cv::Scalar(255));
    for (int i = 0; i < 10; ++i) {
        model->predict(dummy);
    }

    // Step 2: 效能測試
    cv::Mat test_image(480, 640, CV_8UC3);
    cv::randu(test_image, cv::Scalar(0), cv::Scalar(255));

    std::vector<double> latencies;
    for (int i = 0; i < 100; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        model->predict(test_image);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
    }

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    std::cout << "Average latency: " << avg << " ms" << std::endl;
    std::cout << "FPS: " << 1000.0 / avg << std::endl;

    // Note: Runtime 配置（OpenVINO、TensorRT）可透過 Python API 進行
    // C++ 專注於推論執行和效能測試

    return 0;
}
```

**完整範例**：`examples/cpp/embedded_optimization.cpp`

```bash
# 建置與執行
cd build && make embedded_optimization
./embedded_optimization model.onnx --device cuda:0 --benchmark
```

---

### 後端工程師

> **目標**：建立穩定的 AI 推論服務

#### 使用情境

- 需要建立 REST API 推論服務
- 需要監控推論效能和資源使用
- 需要處理高並發請求

#### CLI 工具

```bash
# 查看系統資訊
ivit info

# 列出可用裝置
ivit devices

# 效能測試
ivit benchmark model.onnx --device cuda:0 --iterations 100

# 執行推論
ivit predict model.onnx image.jpg --output result.jpg

# 模型轉換
ivit convert model.onnx model.engine --format tensorrt --fp16

# 啟動推論服務（REST API）
ivit serve model.onnx --port 8080 --device cuda:0

# 模型資訊
ivit export model.onnx --info

# Model Zoo 操作
ivit zoo list
ivit zoo search yolo
ivit zoo download yolov8n
```

#### Callback 系統（監控整合）

```python
import ivit
from ivit.core.callbacks import (
    CallbackManager,
    CallbackContext,
    CallbackEvent,
    FPSCounter,
    LatencyLogger,
    DetectionFilter,
)

# ============================================================
# 情境：建立具備完整監控的推論服務
# ============================================================

# 載入模型（使用 Model Zoo 方便示範）
model = ivit.zoo.load("yolov8n", device="cuda:0")

# 建立 Callback Manager
callback_manager = CallbackManager()

# --- 內建 Callback: FPS 計數器 ---
fps_counter = FPSCounter(window_size=30)
callback_manager.register("infer_end", fps_counter)

# --- 內建 Callback: 延遲記錄器 ---
latency_logger = LatencyLogger()
callback_manager.register("infer_end", latency_logger)

# --- 自定義 Callback: Prometheus 指標 ---
class PrometheusMetricsCallback:
    """將指標發送到 Prometheus"""

    def __init__(self):
        self.inference_count = 0
        self.total_latency = 0

    def __call__(self, ctx: CallbackContext):
        self.inference_count += 1
        self.total_latency += ctx.latency_ms

        # 在實際應用中，這裡會將指標發送到 Prometheus
        # prometheus_client.Counter('inference_total').inc()
        # prometheus_client.Histogram('inference_latency').observe(ctx.latency_ms)

prometheus_callback = PrometheusMetricsCallback()
callback_manager.register("infer_end", prometheus_callback)

# --- 自定義 Callback: 警示系統 ---
def alert_callback(ctx: CallbackContext):
    """延遲過高時發出警示"""
    if ctx.latency_ms > 100:  # 超過 100ms
        print(f"[ALERT] 高延遲警告: {ctx.latency_ms:.1f}ms")
        # 在實際應用中，這裡會發送 Slack/Email 通知

callback_manager.register("infer_end", alert_callback)

# --- 使用 Callback 進行推論 ---
def inference_with_monitoring(image):
    """帶監控的推論函數"""
    # 觸發推論開始事件
    ctx = CallbackContext(event="infer_start", model_name="yolov8n")
    callback_manager.trigger("infer_start", ctx)

    # 執行推論
    import time
    start = time.perf_counter()
    results = model(image)
    latency = (time.perf_counter() - start) * 1000

    # 觸發推論結束事件
    ctx = CallbackContext(
        event="infer_end",
        model_name="yolov8n",
        latency_ms=latency,
        detections=len(results)
    )
    callback_manager.trigger("infer_end", ctx)

    return results

# 執行推論
results = inference_with_monitoring("image.jpg")

# 取得統計資訊
print(f"當前 FPS: {fps_counter.fps:.1f}")
print(f"平均延遲: {latency_logger.average_latency:.1f}ms")
print(f"總推論次數: {prometheus_callback.inference_count}")
```

#### 可用的 Callback 事件

| 事件 | 說明 | Context 屬性 |
|------|------|--------------|
| `pre_process` | 前處理開始 | image_shape |
| `post_process` | 後處理完成 | results |
| `infer_start` | 推論開始 | model_name |
| `infer_end` | 推論結束 | latency_ms, preprocess_ms, inference_ms |
| `batch_start` | 批次開始 | batch_size |
| `batch_end` | 批次結束 | total_latency |
| `stream_start` | 串流開始 | source |
| `stream_frame` | 每一幀 | frame_idx, fps |
| `stream_end` | 串流結束 | total_frames |
| `model_loaded` | 模型載入完成 | model_path, device |
| `model_unloaded` | 模型卸載 | model_path |

#### REST API 服務範例

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import ivit
import numpy as np
import cv2

app = FastAPI(title="iVIT Inference Service")

# 全域模型（應用啟動時載入）
model = None
fps_counter = None

@app.on_event("startup")
async def startup():
    global model, fps_counter
    from ivit.core.callbacks import FPSCounter

    model = ivit.zoo.load("yolov8n", device=ivit.devices.best())
    model.warmup(10)

    fps_counter = FPSCounter(window_size=100)
    model.on("infer_end", fps_counter)

    print(f"模型已載入至 {model.device}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """執行物件偵測"""
    # 讀取圖像
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 推論
    results = model(image)

    return JSONResponse({
        "success": True,
        "detections": results.to_dict()["detections"],
        "inference_time_ms": results.inference_time_ms,
        "current_fps": fps_counter.fps
    })

@app.get("/health")
async def health():
    """健康檢查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "current_fps": fps_counter.fps if fps_counter else 0
    }

@app.get("/stats")
async def stats():
    """效能統計"""
    return {
        "fps": fps_counter.fps,
        "device": str(model.device) if model else None,
    }

# 啟動: uvicorn server:app --host 0.0.0.0 --port 8080
```

#### C++ 範例

```cpp
#include "ivit/ivit.hpp"
#include <opencv2/opencv.hpp>
#include <mutex>
#include <deque>
#include <numeric>

using namespace ivit;

// FPS 計數器（滑動視窗）
class FPSCounter {
public:
    explicit FPSCounter(size_t window_size = 30)
        : window_size_(window_size) {}

    void record(double latency_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        latencies_.push_back(latency_ms);
        while (latencies_.size() > window_size_)
            latencies_.pop_front();
    }

    double fps() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latencies_.empty()) return 0.0;
        double avg_ms = std::accumulate(latencies_.begin(), latencies_.end(), 0.0) / latencies_.size();
        return avg_ms > 0 ? 1000.0 / avg_ms : 0.0;
    }

private:
    size_t window_size_;
    std::deque<double> latencies_;
    mutable std::mutex mutex_;
};

// 推論服務
class InferenceService {
public:
    InferenceService(const std::string& model_path, const DeviceInfo& device)
        : fps_counter_(30) {

        // 使用 load_model API
        LoadConfig config;
        config.device = device.id;
        model_ = load_model(model_path, config);

        // 預熱
        cv::Mat dummy(480, 640, CV_8UC3);
        cv::randu(dummy, cv::Scalar(0), cv::Scalar(255));
        for (int i = 0; i < 10; ++i) {
            model_->predict(dummy);
        }
    }

    Results infer(const cv::Mat& image) {
        auto start = std::chrono::high_resolution_clock::now();
        auto results = model_->predict(image);
        auto end = std::chrono::high_resolution_clock::now();

        double latency = std::chrono::duration<double, std::milli>(end - start).count();
        fps_counter_.record(latency);

        return results;
    }

    double fps() const { return fps_counter_.fps(); }

private:
    std::shared_ptr<Model> model_;
    FPSCounter fps_counter_;
};

int main() {
    auto device = get_best_device();
    InferenceService service("yolov8n.onnx", device);

    cv::Mat image = cv::imread("test.jpg");
    auto results = service.infer(image);

    std::cout << "FPS: " << service.fps() << std::endl;
    std::cout << "Detections: " << results.detections.size() << std::endl;

    // Note: 完整的 Callback 系統可透過 Python API 使用

    return 0;
}
```

**完整範例**：`examples/cpp/backend_service.cpp`

```bash
# 建置與執行
cd build && make backend_service
./backend_service model.onnx --device cuda:0 --demo
```

---

### 資料科學家

> **目標**：快速驗證模型效果，進行實驗分析

#### 使用情境

- 需要快速載入和測試不同模型
- 需要分析推論結果
- 需要將結果匯出為各種格式

#### 快速實驗流程

```python
import ivit

# ============================================================
# 情境：從 Model Zoo 快速載入和測試模型
# ============================================================

# 從 Model Zoo 載入預訓練模型（推薦）
model = ivit.zoo.load("yolov8n", device="cuda:0")

# 執行推論
results = model("image.jpg")

# 快速分析結果
print(f"偵測到 {len(results)} 個物件")
for det in results:
    print(f"  - {det.label}: {det.confidence:.2%}")
```

> **Model Zoo 完整清單**：iVIT 提供 14 個預訓練模型，包含偵測、分類、分割、姿態估計四種任務。
> 詳細的模型列表、效能指標、API 說明請參考 [Model Zoo 完整清單](#model-zoo-完整清單)。

#### Results API 詳解

```python
import ivit
from ivit.core.result import Results

# ============================================================
# 情境：完整的結果處理和分析
# ============================================================

# 載入模型（使用 Model Zoo）
model = ivit.zoo.load("yolov8n")
results = model("image.jpg")

# --- 基本資訊 ---
print(f"偵測數量: {len(results)}")
print(f"推論時間: {results.inference_time_ms:.1f}ms")
print(f"使用裝置: {results.device_used}")
print(f"圖像尺寸: {results.image_size}")

# --- 迭代偵測結果 ---
for det in results:
    print(f"類別: {det.label}")
    print(f"信心度: {det.confidence:.2%}")
    print(f"邊界框: ({det.bbox.x1}, {det.bbox.y1}) - ({det.bbox.x2}, {det.bbox.y2})")
    print(f"面積: {det.bbox.area}")
    print("---")

# --- 過濾功能 ---
# 通用過濾方法
filtered = results.filter(confidence=0.9)
print(f"高信心度偵測: {len(filtered)} 項")

filtered = results.filter(classes=["person", "car"])
print(f"特定類別偵測: {len(filtered)} 項")

filtered = results.filter(confidence=0.8, classes=["person"], min_area=1000)
print(f"組合過濾: {len(filtered)} 項")

# 特定過濾方法
high_conf = results.filter_by_confidence(0.9)
persons = results.filter_by_class(["person"])
large_objects = results.filter_by_area(min_area=5000, max_area=50000)

# --- 序列化 ---
# 轉為字典
data = results.to_dict()

# 轉為 JSON
json_str = results.to_json()

# --- 視覺化 ---
# 顯示結果（阻塞）
results.show()

# 顯示結果（非阻塞）
results.show(wait=False)

# 繪製結果並取得圖像
plotted = results.plot(
    show_labels=True,
    show_confidence=True,
    line_width=2
)

# --- 儲存結果 ---
# 儲存視覺化圖像
results.save("output.jpg")
results.save("output.png")

# 儲存結果資料
results.save("output.json")

# 儲存 YOLO 格式標註
results.save("output.txt")
```

#### 分類結果處理

```python
model = ivit.zoo.load("resnet50")
results = model("cat.jpg")

# 取得 Top-1 結果
top1 = results.top1
print(f"預測類別: {top1.label}")
print(f"信心度: {top1.score:.2%}")

# 取得 Top-5 結果
for i, cls in enumerate(results.top5):
    print(f"{i+1}. {cls.label}: {cls.score:.2%}")

# 取得 Top-K 結果
topk = results.topk(10)
```

#### 分割結果處理

```python
model = ivit.zoo.load("yolov8n-seg")
results = model("street.jpg")

# 取得分割遮罩
mask = results.segmentation_mask  # numpy array

# 上色遮罩
colored_mask = results.colorize_mask()

# 疊加到原圖
overlay = results.overlay_mask(original_image, alpha=0.5)

# 取得輪廓
contours = results.get_contours()
contours_person = results.get_contours(class_id=0)  # 特定類別
```

#### 模型匯出格式比較

```python
from ivit.train import ModelExporter

# 支援的匯出格式
formats = {
    "onnx": {
        "用途": "跨平台部署",
        "優點": "相容性最高",
        "量化": ["fp32", "fp16"],
    },
    "torchscript": {
        "用途": "PyTorch 生態系統",
        "優點": "無需 ONNX 轉換",
        "量化": ["fp32"],
    },
    "openvino": {
        "用途": "Intel 硬體優化",
        "優點": "最佳 Intel 效能",
        "量化": ["fp32", "fp16", "int8"],
    },
    "tensorrt": {
        "用途": "NVIDIA 硬體優化",
        "優點": "最佳 NVIDIA 效能",
        "量化": ["fp32", "fp16", "int8"],
    },
}

# 匯出範例
exporter = ModelExporter(model, device)
exporter.export("model.onnx", format="onnx", quantize="fp16")
```

#### C++ 範例

```cpp
#include "ivit/ivit.hpp"
#include <opencv2/opencv.hpp>
#include <map>
#include <numeric>

using namespace ivit;

int main() {
    // Step 1: 系統探索
    auto devices = list_devices();
    std::cout << "Available devices: " << devices.size() << std::endl;
    for (const auto& dev : devices) {
        std::cout << "  - " << dev.id << ": " << dev.name << std::endl;
    }

    // Note: Model Zoo 可透過 Python API 使用 (ivit.zoo.list_models())

    // Step 2: 載入模型（使用 load_model API）
    auto device = get_best_device();
    LoadConfig config;
    config.device = device.id;
    auto model = load_model("yolov8n.onnx", config);

    // Step 3: 結果分析
    cv::Mat image = cv::imread("test.jpg");
    auto results = model->predict(image);

    std::cout << "Detection count: " << results.detections.size() << std::endl;
    std::cout << "Inference time: " << results.inference_time_ms << " ms" << std::endl;

    // 過濾與統計
    std::map<std::string, int> class_counts;
    int high_conf_count = 0;

    for (const auto& det : results.detections) {
        class_counts[det.label]++;
        if (det.confidence > 0.9) high_conf_count++;

        std::cout << "  " << det.label << ": "
                  << (det.confidence * 100) << "%" << std::endl;
    }

    std::cout << "\nClass distribution:" << std::endl;
    for (const auto& [cls, count] : class_counts) {
        std::cout << "  " << cls << ": " << count << std::endl;
    }
    std::cout << "High confidence (>90%): " << high_conf_count << std::endl;

    // Step 4: 批次處理
    std::vector<cv::Mat> batch_images;
    for (int i = 0; i < 5; ++i) {
        cv::Mat img(480, 640, CV_8UC3);
        cv::randu(img, cv::Scalar(0), cv::Scalar(255));
        batch_images.push_back(img);
    }

    std::vector<double> latencies;
    int total_detections = 0;

    for (const auto& img : batch_images) {
        auto res = model->predict(img);
        latencies.push_back(res.inference_time_ms);
        total_detections += res.detections.size();
    }

    double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    std::cout << "\nBatch stats:" << std::endl;
    std::cout << "  Total detections: " << total_detections << std::endl;
    std::cout << "  Avg latency: " << avg_latency << " ms" << std::endl;

    return 0;
}
```

**完整範例**：`examples/cpp/data_analysis.cpp`

```bash
# 建置與執行
cd build && make data_analysis
./data_analysis model.onnx image.jpg --batch
```

---

## 核心 API 參考

### ivit 模組

```python
import ivit

# 載入模型
model = ivit.load(source, device="auto", task=None)

# 裝置探索
ivit.devices()                    # 列出所有裝置
ivit.devices.cpu()                # 取得 CPU 裝置
ivit.devices.cuda()               # 取得 CUDA 裝置
ivit.devices.npu()                # 取得 NPU 裝置
ivit.devices.best()               # 取得最佳裝置
ivit.devices.best("efficiency")   # 取得最高效率裝置

# 版本資訊
ivit.__version__                  # SDK 版本
ivit.is_cpp_available()           # C++ 綁定是否可用
```

### Model 類別

```python
# 推論
results = model(image)                    # 單張圖像
results = model.predict(image)            # 同上
results = model.predict_batch([img1, img2])  # 批次推論

# 串流推論
for results in model.stream("video.mp4"):
    results.show(wait=False)

# TTA（測試時增強）
results = model.predict_tta(image, scales=[0.8, 1.0, 1.2])

# 配置
model.configure_openvino(...)
model.configure_tensorrt(...)
model.configure_qnn(...)  # Qualcomm IQ Series (規劃中)

# 前後處理
model.set_preprocessor(preprocessor)
model.set_postprocessor(postprocessor)

# 預熱
model.warmup(iterations=10)

# Callback
model.on("infer_end", callback_func)
model.remove_callback("infer_end", callback_func)

# 底層存取
model.runtime                    # Runtime 資訊
model.runtime_handle             # 底層 handle
model.infer_raw(inputs)          # 原始推論
```

### Results 類別

```python
# 基本屬性
len(results)                     # 結果數量
results.inference_time_ms        # 推論時間
results.device_used              # 使用裝置
results.image_size               # 圖像尺寸

# 偵測結果
results.detections               # 所有偵測
results.filter(confidence=0.9)   # 過濾
results.filter_by_class(["person"])
results.filter_by_confidence(0.9)
results.filter_by_area(1000, 50000)

# 分類結果
results.top1                     # Top-1
results.top5                     # Top-5
results.topk(k)                  # Top-K

# 分割結果
results.segmentation_mask        # 分割遮罩
results.colorize_mask()          # 上色
results.overlay_mask(image)      # 疊加
results.get_contours()           # 輪廓

# 序列化
results.to_dict()                # 轉字典
results.to_json()                # 轉 JSON

# 視覺化
results.show()                   # 顯示
results.plot()                   # 繪製
results.save(path)               # 儲存
```

---

## 最佳實務

### 效能優化

1. **使用正確的裝置**
   ```python
   # 自動選擇最佳裝置
   model = ivit.zoo.load("yolov8n", device=ivit.devices.best())
   # 或使用本地檔案
   # model = ivit.load("model.onnx", device=ivit.devices.best())
   ```

2. **執行預熱**
   ```python
   model.warmup(iterations=10)
   ```

3. **使用 FP16 量化**
   ```python
   model.configure_tensorrt(fp16=True)
   ```

4. **批次推論**
   ```python
   results = model.predict_batch([img1, img2, img3, img4])
   ```

### 記憶體管理

1. **及時釋放模型**
   ```python
   del model
   ```

2. **使用串流模式處理影片**
   ```python
   for results in model.stream("video.mp4"):
       process(results)
   ```

### 錯誤處理

1. **總是包裝 try-except**
   ```python
   try:
       results = model(image)
   except ivit.IVITError as e:
       logger.error(f"推論失敗: {e}")
   ```

2. **驗證輸入**
   ```python
   if image is None or image.size == 0:
       raise ValueError("無效的輸入圖像")
   ```

---

## 故障排除

### 常見問題

#### Q: 找不到 CUDA 裝置

```
DeviceNotFoundError: CUDA device not found
```

**解決方案**：
1. 確認 NVIDIA 驅動程式已安裝：`nvidia-smi`
2. 確認 CUDA toolkit 已安裝
3. 執行 `ivit.devices()` 確認可用裝置

#### Q: 模型載入失敗

```
ModelLoadError: Failed to load model
```

**解決方案**：
1. 確認模型檔案路徑正確
2. 確認模型格式支援（.onnx, .xml, .engine）
3. 確認對應的後端已安裝

#### Q: 推論結果異常

**解決方案**：
1. 確認輸入圖像格式正確（BGR vs RGB）
2. 確認前處理參數與訓練時一致
3. 檢查模型輸入尺寸要求

#### Q: 效能不如預期

**解決方案**：
1. 執行 `model.warmup(10)` 預熱
2. 啟用 FP16 量化
3. 使用 `ivit benchmark` 進行效能測試
4. 檢查是否有其他程式佔用 GPU

---

## Model Zoo 完整清單

iVIT Model Zoo 提供 14 個預訓練模型，涵蓋四種電腦視覺任務。

### 物件偵測 (Detection)

| 模型名稱 | 輸入尺寸 | mAP50-95 | 參數量 | FLOPs | 適用場景 |
|----------|----------|----------|--------|-------|----------|
| `yolov8n` | 640×640 | 37.3% | 3.2M | 8.7G | 邊緣裝置、即時偵測 |
| `yolov8s` | 640×640 | 44.9% | 11.2M | 28.6G | 平衡速度與準確率 |
| `yolov8m` | 640×640 | 50.2% | 25.9M | 78.9G | 中等準確率需求 |
| `yolov8l` | 640×640 | 52.9% | 43.7M | 165.2G | 高準確率需求 |
| `yolov8x` | 640×640 | 53.9% | 68.2M | 257.8G | 最高準確率、伺服器部署 |

### 圖像分類 (Classification)

| 模型名稱 | 輸入尺寸 | Top-1 | Top-5 | 來源 | 適用場景 |
|----------|----------|-------|-------|------|----------|
| `yolov8n-cls` | 224×224 | 69.0% | 88.3% | Ultralytics | 邊緣裝置分類 |
| `yolov8s-cls` | 224×224 | 73.8% | 91.7% | Ultralytics | 平衡效能分類 |
| `resnet50` | 224×224 | 76.1% | 92.9% | TorchVision | 經典分類模型 |
| `mobilenetv3` | 224×224 | 74.0% | 91.3% | TorchVision | 行動裝置分類 |
| `efficientnet-b0` | 224×224 | 77.1% | 93.3% | TorchVision | 高效率分類 |

### 語意分割 (Segmentation)

| 模型名稱 | 輸入尺寸 | mAP (Box) | mAP (Mask) | 適用場景 |
|----------|----------|-----------|------------|----------|
| `yolov8n-seg` | 640×640 | 36.7% | 30.5% | 邊緣裝置分割 |
| `yolov8s-seg` | 640×640 | 44.6% | 36.8% | 平衡效能分割 |

### 姿態估計 (Pose Estimation)

| 模型名稱 | 輸入尺寸 | mAP (Pose) | 適用場景 |
|----------|----------|------------|----------|
| `yolov8n-pose` | 640×640 | 50.4% | 邊緣裝置姿態 |
| `yolov8s-pose` | 640×640 | 60.0% | 平衡效能姿態 |

### Model Zoo API 參考

> **基本載入方式**請參考 [30 秒快速體驗](#30-秒快速體驗)

```python
import ivit

# ===== 瀏覽模型 =====

# 列出所有模型
models = ivit.zoo.list_models()
# ['efficientnet-b0', 'mobilenetv3', 'resnet50', 'yolov8l', ...]

# 按任務過濾
ivit.zoo.list_models(task="detect")    # 偵測模型 (5)
ivit.zoo.list_models(task="classify")  # 分類模型 (5)
ivit.zoo.list_models(task="segment")   # 分割模型 (2)
ivit.zoo.list_models(task="pose")      # 姿態模型 (2)

# ===== 搜尋模型 =====

ivit.zoo.search("yolo")     # 搜尋名稱含 "yolo" 的模型
ivit.zoo.search("edge")     # 搜尋標籤含 "edge" 的模型
ivit.zoo.search("fast")     # 搜尋標籤含 "fast" 的模型

# ===== 查詢模型資訊 =====

info = ivit.zoo.get_model_info("yolov8n")
print(f"任務: {info.task}")           # detect
print(f"輸入尺寸: {info.input_size}") # (640, 640)
print(f"類別數: {info.num_classes}")  # 80
print(f"效能指標: {info.metrics}")    # {'mAP50-95': 37.3, 'params_m': 3.2, ...}
print(f"標籤: {info.tags}")           # ['yolo', 'detection', 'fast', 'edge']

# ===== 載入模型（指定裝置）=====

model = ivit.zoo.load("yolov8n", device="cuda:0")   # 指定 GPU
model = ivit.zoo.load("yolov8n", device="cpu")      # 指定 CPU
model = ivit.zoo.load("yolov8n", device="npu")      # 指定 NPU
```

### 模型選擇建議

| 場景 | 推薦模型 | 理由 |
|------|----------|------|
| 邊緣裝置即時偵測 | `yolov8n` | 最小、最快 |
| 一般偵測應用 | `yolov8s` | 平衡速度與準確率 |
| 高準確率需求 | `yolov8m` 或 `yolov8l` | 準確率更高 |
| 行動裝置分類 | `mobilenetv3` | 專為行動裝置優化 |
| 高精度分類 | `efficientnet-b0` | Top-1 最高 |
| 人體姿態追蹤 | `yolov8n-pose` | 即時姿態估計 |

---

## 附錄

### A. 支援的模型格式

| 格式 | 副檔名 | 說明 |
|------|--------|------|
| ONNX | .onnx | 開放神經網路交換格式 |
| OpenVINO IR | .xml, .bin | Intel 優化格式 |
| TensorRT Engine | .engine, .trt | NVIDIA 優化格式 |
| TorchScript | .pt, .pth | PyTorch 格式 |

### B. 環境變數

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `IVIT_CACHE_DIR` | 模型快取目錄 | `~/.ivit/cache` |
| `IVIT_LOG_LEVEL` | 日誌等級 | `INFO` |
| `IVIT_DEFAULT_DEVICE` | 預設裝置 | `auto` |

### C. 相關連結

- [GitHub Repository](https://github.com/innodisk-mannywang/ivit-sdk)
- [API 文件](./api/api-spec.md)
- [架構設計](./architecture/adr-001-system.md)
- [產品需求](./development/prd.md)

---

## 版本歷史

| 版本 | 日期 | 說明 |
|------|------|------|
| 1.0.0 | 2026-01-26 | 初始版本 |

---

> **需要協助？** 請前往 [GitHub Issues](https://github.com/innodisk-mannywang/ivit-sdk/issues) 提交問題。
