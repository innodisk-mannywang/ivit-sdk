#!/bin/bash
# iVIT-SDK Build Test Script
#
# 此腳本在 Docker 容器中執行，驗證 SDK 編譯

set -e

echo "=============================================="
echo "  iVIT-SDK Clean Build Verification"
echo "=============================================="
echo ""

# 檢查原始碼是否掛載
if [ ! -d "/workspace/ivit-sdk" ]; then
    echo "ERROR: Source code not found at /workspace/ivit-sdk"
    echo "Please mount the source directory:"
    echo "  docker run --rm -v \$(pwd):/workspace/ivit-sdk:ro <image>"
    exit 1
fi

# 複製到可寫目錄
echo "[1/5] Copying source code..."
cp -r /workspace/ivit-sdk /tmp/ivit-sdk
cd /tmp/ivit-sdk

# 清理並建立 build 目錄
echo "[2/5] Configuring CMake..."
rm -rf build && mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_USE_OPENVINO=OFF \
    -DIVIT_USE_TENSORRT=OFF \
    -DIVIT_USE_ONNXRUNTIME=OFF \
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

# 執行測試
echo ""
echo "[5/5] Running tests..."
ctest --output-on-failure

echo ""
echo "=============================================="
echo "  Build Verification PASSED"
echo "=============================================="
