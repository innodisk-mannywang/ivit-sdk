#!/bin/bash
# iVIT-SDK Full Build Test Script (All backends ON)

set -e

echo "=============================================="
echo "  iVIT-SDK Full Build Verification"
echo "  (OpenVINO + TensorRT)"
echo "=============================================="
echo ""

# 檢查原始碼
if [ ! -d "/workspace/ivit-sdk" ]; then
    echo "ERROR: Source code not found at /workspace/ivit-sdk"
    exit 1
fi

# 載入 OpenVINO 環境
echo "[0/5] Loading OpenVINO environment..."
source /opt/intel/openvino_2024/setupvars.sh 2>/dev/null || source /opt/intel/openvino/setupvars.sh

# 複製原始碼
echo "[1/5] Copying source code..."
cp -r /workspace/ivit-sdk /tmp/ivit-sdk
cd /tmp/ivit-sdk

# CMake 配置
echo "[2/5] Configuring CMake (all backends ON)..."
rm -rf build && mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_USE_OPENVINO=ON \
    -DIVIT_USE_TENSORRT=ON \
    -DIVIT_BUILD_TESTS=ON \
    -DIVIT_BUILD_EXAMPLES=ON \
    -DIVIT_BUILD_PYTHON=ON

# 編譯
echo ""
echo "[3/5] Building..."
make -j$(nproc)

# 顯示產物
echo ""
echo "[4/5] Build artifacts:"
echo "  Libraries:"
ls -la lib/
echo ""
echo "  Binaries:"
ls -la bin/

# 測試
echo ""
echo "[5/5] Running tests..."
ctest --output-on-failure

echo ""
echo "=============================================="
echo "  Full Build Verification PASSED"
echo "=============================================="
