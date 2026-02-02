# Clean Build Verification Guide

本文件說明如何在乾淨的 Docker 環境中驗證 iVIT-SDK 的 C++ 編譯流程。

## 目的

確保專案在全新環境下（無任何預裝依賴）能夠成功編譯，驗證：
- 所有依賴都有正確宣告
- CMakeLists.txt 配置正確
- 編譯流程可重現

## 前置需求

- Docker 已安裝
- iVIT-SDK 原始碼

## 基本編譯驗證（不含推論後端）

### 1. 建立 Dockerfile

```bash
cat > /tmp/ivit-cpp-clean.Dockerfile << 'EOF'
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libgtest-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pybind11

WORKDIR /workspace
EOF
```

### 2. 建立 Docker 映像

```bash
docker build -t ivit-clean-test -f /tmp/ivit-cpp-clean.Dockerfile .
```

### 3. 啟動容器

```bash
docker run -it --rm \
    -v /path/to/ivit-sdk:/workspace/ivit-sdk:ro \
    ivit-clean-test bash
```

### 4. 在容器內編譯

```bash
# 複製到可寫目錄
cp -r /workspace/ivit-sdk /tmp/ivit-sdk
cd /tmp/ivit-sdk

# 清理並編譯
rm -rf build && mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_USE_OPENVINO=OFF \
    -DIVIT_USE_TENSORRT=OFF

make -j$(nproc)

# 執行測試
ctest -V

# 驗證產物
ls -la lib/ bin/
```

## 含 OpenVINO 後端驗證

### 1. 建立 Dockerfile

```bash
cat > /tmp/ivit-openvino.Dockerfile << 'EOF'
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    gpg-agent \
    libopencv-dev \
    libgtest-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安裝 OpenVINO 2024
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor -o /usr/share/keyrings/intel.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/intel.gpg] https://apt.repos.intel.com/openvino/2024 ubuntu22 main" \
    > /etc/apt/sources.list.d/intel-openvino.list \
    && apt-get update && apt-get install -y openvino-2024.0.0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pybind11

WORKDIR /workspace
EOF
```

### 2. 建立映像並啟動

```bash
docker build -t ivit-openvino-test -f /tmp/ivit-openvino.Dockerfile .

docker run -it --rm \
    -v /path/to/ivit-sdk:/workspace/ivit-sdk:ro \
    ivit-openvino-test bash
```

### 3. 在容器內編譯

```bash
# 載入 OpenVINO 環境
source /opt/intel/openvino/setupvars.sh

# 複製並編譯
cp -r /workspace/ivit-sdk /tmp/ivit-sdk
cd /tmp/ivit-sdk
rm -rf build && mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_USE_OPENVINO=ON \
    -DIVIT_USE_TENSORRT=OFF

make -j$(nproc)
ctest -V
```

## 含 TensorRT 後端驗證

### 1. 建立 Dockerfile

```bash
cat > /tmp/ivit-tensorrt.Dockerfile << 'EOF'
FROM nvcr.io/nvidia/tensorrt:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pybind11

WORKDIR /workspace
EOF
```

### 2. 建立映像並啟動（需要 NVIDIA GPU）

```bash
docker build -t ivit-tensorrt-test -f /tmp/ivit-tensorrt.Dockerfile .

docker run -it --rm --gpus all \
    -v /path/to/ivit-sdk:/workspace/ivit-sdk:ro \
    ivit-tensorrt-test bash
```

### 3. 在容器內編譯

```bash
cp -r /workspace/ivit-sdk /tmp/ivit-sdk
cd /tmp/ivit-sdk
rm -rf build && mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_USE_OPENVINO=OFF \
    -DIVIT_USE_TENSORRT=ON

make -j$(nproc)
ctest -V
```

## 完整後端驗證

若需同時驗證多個後端，可以組合上述 Dockerfile 或使用 `deps/` 目錄中的預編譯依賴：

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_BUNDLE_DEPS=ON \
    -DIVIT_USE_OPENVINO=ON \
    -DIVIT_USE_TENSORRT=ON
```

## 預期結果

成功編譯後應該看到：

```
lib/
├── libivit.so

bin/
├── simple_inference
├── classification_demo
├── detection_demo
├── video_demo
├── si_quickstart
├── embedded_optimization
├── backend_service
├── data_analysis
├── test_model
└── test_inference
```

## 常見問題

### Q: CMake 找不到 OpenCV
確認已安裝 `libopencv-dev`：
```bash
apt-get install libopencv-dev
```

### Q: pybind11 找不到
確認已安裝：
```bash
pip3 install pybind11
```

### Q: OpenVINO 找不到
確認已執行環境設定：
```bash
source /opt/intel/openvino/setupvars.sh
```

### Q: TensorRT 找不到 CUDA
確認使用 `--gpus all` 啟動容器，並使用 NVIDIA 官方 TensorRT 映像。
