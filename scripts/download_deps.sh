#!/bin/bash
# ============================================================================
# download_deps.sh
# 下載並準備 iVIT-SDK 依賴庫
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(dirname "$SCRIPT_DIR")"
DEPS_DIR="${SDK_ROOT}/deps"
INSTALL_DIR="${DEPS_DIR}/install"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 檢測系統架構
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    ARCH_SUFFIX="x64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH_SUFFIX="aarch64"
else
    log_error "Unsupported architecture: $ARCH"
    exit 1
fi

log_info "Detected architecture: $ARCH ($ARCH_SUFFIX)"

# 建立目錄
mkdir -p "$DEPS_DIR"
mkdir -p "$INSTALL_DIR"

# ============================================================================
# 下載 ONNX Runtime
# ============================================================================
download_onnxruntime() {
    local VERSION="1.17.0"
    local GPU_SUFFIX=""

    # 檢查是否有 CUDA
    if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
        GPU_SUFFIX="-gpu"
        log_info "CUDA detected, downloading GPU version"
    fi

    local URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-${ARCH_SUFFIX}${GPU_SUFFIX}-${VERSION}.tgz"
    local ARCHIVE="${DEPS_DIR}/onnxruntime-${VERSION}.tgz"

    if [ -d "${INSTALL_DIR}/onnxruntime" ]; then
        log_info "ONNX Runtime already exists, skipping..."
        return
    fi

    log_info "Downloading ONNX Runtime ${VERSION}..."
    wget -q --show-progress -O "$ARCHIVE" "$URL"

    log_info "Extracting ONNX Runtime..."
    mkdir -p "${INSTALL_DIR}/onnxruntime"
    tar -xzf "$ARCHIVE" -C "${INSTALL_DIR}/onnxruntime" --strip-components=1

    rm -f "$ARCHIVE"
    log_info "ONNX Runtime installed to ${INSTALL_DIR}/onnxruntime"
}

# ============================================================================
# 下載 OpenVINO
# ============================================================================
download_openvino() {
    local VERSION="2024.0.0"
    local OS_SUFFIX="ubuntu22"  # ubuntu20, ubuntu22, rhel8

    # 檢測 Ubuntu 版本
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$VERSION_ID" == "20.04" ]]; then
            OS_SUFFIX="ubuntu20"
        elif [[ "$VERSION_ID" == "22.04" ]]; then
            OS_SUFFIX="ubuntu22"
        fi
    fi

    local URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/${VERSION}/linux/l_openvino_toolkit_${OS_SUFFIX}_${VERSION}.10926.b5b0aea2895_${ARCH_SUFFIX}.tgz"
    local ARCHIVE="${DEPS_DIR}/openvino-${VERSION}.tgz"

    if [ -d "${INSTALL_DIR}/runtime" ]; then
        log_info "OpenVINO already exists, skipping..."
        return
    fi

    log_info "Downloading OpenVINO ${VERSION}..."
    wget -q --show-progress -O "$ARCHIVE" "$URL" || {
        log_warn "Failed to download from official URL, trying alternative..."
        # 嘗試替代 URL
        URL="https://github.com/openvinotoolkit/openvino/releases/download/${VERSION}/l_openvino_toolkit_${OS_SUFFIX}_${VERSION}_${ARCH_SUFFIX}.tgz"
        wget -q --show-progress -O "$ARCHIVE" "$URL" || {
            log_error "Failed to download OpenVINO"
            return 1
        }
    }

    log_info "Extracting OpenVINO..."
    tar -xzf "$ARCHIVE" -C "${INSTALL_DIR}" --strip-components=1

    rm -f "$ARCHIVE"
    log_info "OpenVINO installed to ${INSTALL_DIR}"
}

# ============================================================================
# 設定 TensorRT (需要手動下載)
# ============================================================================
setup_tensorrt() {
    log_warn "TensorRT requires manual download from NVIDIA Developer website"
    log_info "Please download TensorRT from: https://developer.nvidia.com/tensorrt"
    log_info "Then extract to: ${INSTALL_DIR}/tensorrt"

    cat << 'EOF'

TensorRT 安裝步驟:
1. 前往 https://developer.nvidia.com/tensorrt
2. 登入 NVIDIA Developer 帳號
3. 下載對應 CUDA 版本的 TensorRT (推薦 tar 格式)
4. 解壓縮到 deps/install/tensorrt:
   tar -xzf TensorRT-*.tgz
   mv TensorRT-* deps/install/tensorrt

EOF

    # 建立目錄結構提示
    mkdir -p "${INSTALL_DIR}/tensorrt"
    cat > "${INSTALL_DIR}/tensorrt/README.txt" << 'EOF'
Please download and extract TensorRT here.

Required files:
  include/NvInfer.h
  lib/libnvinfer.so*
  lib/libnvonnxparser.so*

Download from: https://developer.nvidia.com/tensorrt
EOF
}

# ============================================================================
# 驗證安裝
# ============================================================================
verify_installation() {
    log_info "Verifying installation..."

    local has_error=0

    # 檢查 ONNX Runtime
    if [ -f "${INSTALL_DIR}/onnxruntime/lib/libonnxruntime.so" ]; then
        log_info "✓ ONNX Runtime: OK"
    else
        log_warn "✗ ONNX Runtime: Not found"
        has_error=1
    fi

    # 檢查 OpenVINO
    if [ -f "${INSTALL_DIR}/runtime/lib/intel64/libopenvino.so" ] || \
       [ -f "${INSTALL_DIR}/runtime/lib/libopenvino.so" ]; then
        log_info "✓ OpenVINO: OK"
    else
        log_warn "✗ OpenVINO: Not found"
        has_error=1
    fi

    # 檢查 TensorRT
    if [ -f "${INSTALL_DIR}/tensorrt/lib/libnvinfer.so" ]; then
        log_info "✓ TensorRT: OK"
    else
        log_warn "✗ TensorRT: Not found (optional, requires manual download)"
    fi

    return $has_error
}

# ============================================================================
# 建立 CMake 設定檔
# ============================================================================
create_cmake_config() {
    log_info "Creating CMake configuration..."

    cat > "${DEPS_DIR}/deps_config.cmake" << EOF
# Auto-generated dependency configuration
# Generated by download_deps.sh

set(IVIT_DEPS_INSTALL_DIR "${INSTALL_DIR}" CACHE PATH "Dependencies install directory" FORCE)

# ONNX Runtime
if(EXISTS "${INSTALL_DIR}/onnxruntime/lib/libonnxruntime.so")
    set(ONNXRUNTIME_ROOT "${INSTALL_DIR}/onnxruntime" CACHE PATH "ONNX Runtime root" FORCE)
    set(ONNXRUNTIME_INCLUDE_DIR "${INSTALL_DIR}/onnxruntime/include" CACHE PATH "ONNX Runtime include" FORCE)
    set(ONNXRUNTIME_LIBRARY "${INSTALL_DIR}/onnxruntime/lib/libonnxruntime.so" CACHE FILEPATH "ONNX Runtime library" FORCE)
endif()

# OpenVINO
if(EXISTS "${INSTALL_DIR}/runtime/cmake/OpenVINOConfig.cmake")
    set(OpenVINO_DIR "${INSTALL_DIR}/runtime/cmake" CACHE PATH "OpenVINO CMake dir" FORCE)
endif()

# TensorRT
if(EXISTS "${INSTALL_DIR}/tensorrt/lib/libnvinfer.so")
    set(TENSORRT_ROOT "${INSTALL_DIR}/tensorrt" CACHE PATH "TensorRT root" FORCE)
endif()

message(STATUS "Loaded bundled dependencies from: ${INSTALL_DIR}")
EOF

    log_info "CMake configuration saved to: ${DEPS_DIR}/deps_config.cmake"
}

# ============================================================================
# 主程式
# ============================================================================
main() {
    log_info "========================================="
    log_info "iVIT-SDK Dependencies Downloader"
    log_info "========================================="

    local download_onnx=1
    local download_ov=1
    local download_trt=0

    # 解析參數
    while [[ $# -gt 0 ]]; do
        case $1 in
            --onnxruntime-only)
                download_ov=0
                download_trt=0
                shift
                ;;
            --openvino-only)
                download_onnx=0
                download_trt=0
                shift
                ;;
            --all)
                download_onnx=1
                download_ov=1
                download_trt=1
                shift
                ;;
            --tensorrt)
                download_trt=1
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --onnxruntime-only  Only download ONNX Runtime"
                echo "  --openvino-only     Only download OpenVINO"
                echo "  --tensorrt          Show TensorRT setup instructions"
                echo "  --all               Download all dependencies"
                echo "  --help              Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # 下載依賴
    if [ $download_onnx -eq 1 ]; then
        download_onnxruntime
    fi

    if [ $download_ov -eq 1 ]; then
        download_openvino
    fi

    if [ $download_trt -eq 1 ]; then
        setup_tensorrt
    fi

    # 建立設定檔
    create_cmake_config

    # 驗證
    verify_installation

    log_info "========================================="
    log_info "Dependencies download complete!"
    log_info ""
    log_info "To build with bundled dependencies, run:"
    log_info "  mkdir build && cd build"
    log_info "  cmake .. -DIVIT_BUNDLE_DEPS=ON -C ../deps/deps_config.cmake"
    log_info "  make -j\$(nproc)"
    log_info "========================================="
}

main "$@"
