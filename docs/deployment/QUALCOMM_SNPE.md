# Qualcomm SNPE 部署指南

本指南說明如何在 Qualcomm 平台上使用 iVIT-SDK 進行模型部署。

## 支援硬體

| 硬體類型 | 架構 | 說明 |
|----------|------|------|
| Qualcomm Hexagon DSP | ARM64 | Snapdragon 8/7/6 系列 |
| Qualcomm Hexagon NPU | ARM64 | Snapdragon 8 Gen 2/3 |
| Qualcomm Adreno GPU | ARM64 | 所有 Snapdragon 平台 |

## 環境準備

### 1. 安裝 SNPE SDK

從 Qualcomm 開發者網站下載 SNPE SDK：
https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk

```bash
# 解壓縮 SNPE SDK
unzip snpe-2.x.x.zip -d /opt/

# 設定環境變數
export SNPE_ROOT=/opt/snpe-2.x.x
export PATH=$SNPE_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/aarch64-android:$LD_LIBRARY_PATH

# 驗證安裝
snpe-net-run --version
```

### 2. 安裝 iVIT-SDK

```bash
pip install ivit-sdk
```

### 3. 確認安裝

```bash
ivit info
```

預期輸出：
```
iVIT-SDK v1.0.0
============================================================
System: Linux (Android 13)
Python: 3.10.12

Available Backends:
  - SNPE 2.18.0 ✓
  - ONNX Runtime 1.16.0 ✓

Devices:
  - dsp: Qualcomm Hexagon DSP (SNPE)
  - npu: Qualcomm Hexagon NPU (SNPE)
  - gpu: Qualcomm Adreno 740 (SNPE)
  - cpu: Qualcomm Kryo (SNPE)
```

## 快速開始

### 基本推論

```python
import ivit

# 載入 DLC 模型
model = ivit.load("model.dlc", device="dsp")

# 執行推論
results = model("image.jpg")
results.show()
```

### 指定裝置

```python
# 使用 Hexagon DSP（最佳效能）
model = ivit.load("model.dlc", device="dsp")

# 使用 Hexagon NPU（較新平台）
model = ivit.load("model.dlc", device="npu")

# 使用 Adreno GPU
model = ivit.load("model.dlc", device="gpu")

# 使用 CPU
model = ivit.load("model.dlc", device="cpu")
```

## 模型轉換

### ONNX 轉 DLC

```bash
# 使用 SNPE 轉換工具
snpe-onnx-to-dlc --input_network model.onnx \
                  --output_path model.dlc

# 或使用 iVIT CLI
ivit convert model.onnx -f snpe -o ./output/
```

### 量化（INT8）

```bash
# 準備量化資料
snpe-dlc-quantize --input_dlc model.dlc \
                  --input_list input_list.txt \
                  --output_dlc model_quantized.dlc
```

## 效能優化

### 配置 SNPE 參數

```python
import ivit

model = ivit.load("model.dlc", device="dsp")

# 配置 SNPE 特定參數
model.configure_snpe(
    runtime="dsp",                           # dsp, npu, gpu, cpu
    performance_profile="HIGH_PERFORMANCE",  # 效能模式
    use_user_supplied_buffers=True,          # 零複製優化
)

# 執行推論
results = model("image.jpg")
```

### 效能模式說明

| 模式 | 說明 | 功耗 | 適用場景 |
|------|------|------|----------|
| DEFAULT | 平衡 | 中 | 一般應用 |
| BALANCED | 效能/功耗平衡 | 中 | 持續運行 |
| HIGH_PERFORMANCE | 最高效能 | 高 | 即時處理 |
| POWER_SAVER | 省電 | 低 | 背景任務 |
| SUSTAINED_HIGH_PERFORMANCE | 持續高效能 | 高 | 長時間高負載 |

### DSP vs NPU 選擇

| 特性 | Hexagon DSP | Hexagon NPU |
|------|-------------|-------------|
| 可用平台 | 大多數 Snapdragon | 8 Gen 2/3 以上 |
| 效能 | 高 | 極高 |
| 功耗 | 中 | 低 |
| 支援運算 | 廣泛 | AI 優化 |

```python
# 自動選擇最佳 Qualcomm 加速器
device = ivit.devices.best()  # 會優先選擇 NPU > DSP > GPU
model = ivit.load("model.dlc", device=device)
```

## 零複製推論

使用 User-Supplied Buffers 減少記憶體複製：

```python
import ivit
import numpy as np

model = ivit.load("model.dlc", device="dsp")

model.configure_snpe(
    use_user_supplied_buffers=True,
)

# 預先分配輸入/輸出 buffer
input_buffer = np.zeros((1, 224, 224, 3), dtype=np.float32)
output_buffer = np.zeros((1, 1000), dtype=np.float32)

# 直接使用 buffer 推論
results = model.infer_raw({
    "input": input_buffer,
})
```

## 串流處理

```python
import ivit

model = ivit.load("model.dlc", device="dsp")

model.configure_snpe(
    performance_profile="SUSTAINED_HIGH_PERFORMANCE",
)

# 處理相機串流
for results in model.stream("/dev/video0"):
    frame = results.plot()
    # 顯示或儲存結果
```

## 效能監控

```python
import ivit
from ivit.core.callbacks import FPSCounter, LatencyLogger

model = ivit.load("model.dlc", device="dsp")

# 開啟 SNPE profiling
model.configure_snpe(
    enable_profiling=True,
    profiling_level="DETAILED",
)

# 新增效能監控
fps = FPSCounter(window_size=30)
model.on("infer_end", fps)

# 執行推論
for i in range(100):
    results = model("image.jpg")

print(f"Average FPS: {fps.fps:.1f}")
```

## Android 部署

### NDK 整合

```cpp
// C++ 範例
#include <ivit/ivit.hpp>

int main() {
    // 載入模型
    auto model = ivit::load("model.dlc", "dsp");

    // 配置 SNPE
    ivit::SNPEConfig config;
    config.runtime = "dsp";
    config.performance_profile = "HIGH_PERFORMANCE";
    model->configure(config);

    // 推論
    auto results = model->predict(image);
    return 0;
}
```

### Gradle 設定

```gradle
android {
    defaultConfig {
        externalNativeBuild {
            cmake {
                arguments "-DANDROID_STL=c++_shared"
            }
        }
    }
}

dependencies {
    implementation 'com.qualcomm:snpe:2.18.0'
}
```

## 常見問題

### Q: 如何確認 DSP/NPU 是否可用？

```bash
# 使用 iVIT
ivit devices

# 使用 SNPE
snpe-platform-validator --runtime dsp
```

### Q: DLC 模型載入失敗？

1. 確認 SNPE 版本相容
2. 確認模型支援目標 runtime
3. 檢查運算元支援

```bash
# 驗證模型
snpe-dlc-info -i model.dlc
```

### Q: 推論速度不如預期？

1. 使用 HIGH_PERFORMANCE 模式
2. 使用量化模型（INT8）
3. 開啟零複製（user-supplied buffers）
4. 減少輸入尺寸

```python
model.configure_snpe(
    performance_profile="HIGH_PERFORMANCE",
    use_user_supplied_buffers=True,
)
```

### Q: 某些層不支援 DSP？

SNPE 會自動回退到 CPU。可以檢查：
```bash
snpe-dlc-info -i model.dlc | grep "Runtime"
```

## 範例程式

完整範例程式位於：`examples/python/snpe_demo.py`

```python
#!/usr/bin/env python3
"""Qualcomm SNPE 部署範例"""

import ivit
from ivit.core.callbacks import FPSCounter

def main():
    # 列出可用裝置
    print("Available Qualcomm devices:")
    for d in ivit.devices():
        if d.backend == "snpe":
            print(f"  - {d.id}: {d.name}")

    # 載入模型
    model = ivit.load("yolov8n.dlc", device="dsp")

    # 配置 SNPE
    model.configure_snpe(
        performance_profile="HIGH_PERFORMANCE",
    )

    # 新增 FPS 計數器
    fps = FPSCounter()
    model.on("infer_end", fps)

    # 預熱
    model.warmup(5)

    # 執行效能測試
    print("\nRunning benchmark on DSP...")
    for i in range(100):
        results = model("image.jpg")

    print(f"\nAverage FPS: {fps.fps:.1f}")
    print(f"Latency: {1000/fps.fps:.1f}ms")

if __name__ == "__main__":
    main()
```
