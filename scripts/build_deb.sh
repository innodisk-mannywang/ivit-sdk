#!/bin/bash
# build_deb.sh — 編譯 + 產出 .deb
#
# Usage: ./scripts/build_deb.sh <platform>
#   platform: x86_64-nvidia | x86_64-intel | aarch64-jetson
#
# Output: build/ivit-sdk_<version>_<arch>-<backend>.deb

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PLATFORM="${1:-}"
if [[ -z "$PLATFORM" ]]; then
    echo "Usage: $0 <platform>"
    echo "  Platforms: x86_64-nvidia | x86_64-intel | aarch64-jetson"
    exit 1
fi

# 設定 CMake 選項 — .deb 不 bundle 依賴
case "$PLATFORM" in
    x86_64-nvidia)
        CMAKE_OPTS="-DIVIT_USE_OPENVINO=OFF -DIVIT_USE_TENSORRT=ON"
        ;;
    x86_64-intel)
        CMAKE_OPTS="-DIVIT_USE_OPENVINO=ON -DIVIT_USE_TENSORRT=OFF"
        ;;
    aarch64-jetson)
        CMAKE_OPTS="-DIVIT_USE_OPENVINO=OFF -DIVIT_USE_TENSORRT=ON"
        ;;
    *)
        echo "Error: Unknown platform '$PLATFORM'"
        exit 1
        ;;
esac

BUILD_DIR="$PROJECT_ROOT/build-deb-$PLATFORM"

echo "=== Building .deb for $PLATFORM ==="

mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_BUNDLE_DEPS=OFF \
    -DIVIT_BUILD_TESTS=OFF \
    -DIVIT_BUILD_EXAMPLES=OFF \
    -DIVIT_BUILD_PYTHON=OFF \
    $CMAKE_OPTS

cmake --build "$BUILD_DIR" -j"$(nproc)"

# 使用 CPack 產生 .deb
cd "$BUILD_DIR"
cpack -G DEB

echo ""
echo "=== .deb package created ==="
ls -lh "$BUILD_DIR"/*.deb 2>/dev/null || echo "Warning: No .deb found"
