#!/bin/bash
# verify_package.sh — 驗證各方案產出是否可用
#
# Usage: ./scripts/verify_package.sh <type> <path>
#   type: tarball | deb | docker
#   path: tarball 路徑 / .deb 路徑 / docker image tag

set -euo pipefail

TYPE="${1:-}"
TARGET="${2:-}"

if [[ -z "$TYPE" || -z "$TARGET" ]]; then
    echo "Usage: $0 <type> <path>"
    echo "  Types: tarball | deb | docker"
    echo ""
    echo "Examples:"
    echo "  $0 tarball build/ivit-sdk-1.0.0-x86_64-nvidia.tar.gz"
    echo "  $0 deb build/ivit-sdk_1.0.0_amd64-nvidia.deb"
    echo "  $0 docker innodisk/ivit-sdk:1.0.0-nvidia"
    exit 1
fi

PASS=0
FAIL=0

check() {
    local desc="$1"
    shift
    echo -n "  [$desc] ... "
    if "$@" >/dev/null 2>&1; then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL"
        ((FAIL++))
    fi
}

verify_tarball() {
    local tarball="$1"
    local tmpdir
    tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" EXIT

    echo "=== Verifying tarball: $tarball ==="

    check "Extract tarball" tar -xzf "$tarball" -C "$tmpdir"

    local sdk_dir
    sdk_dir=$(ls -d "$tmpdir"/ivit-sdk-* 2>/dev/null | head -1)
    if [[ -z "$sdk_dir" ]]; then
        echo "  FAIL: No SDK directory found after extraction"
        ((FAIL++))
        return
    fi

    check "libivit.so exists" test -f "$sdk_dir/lib/libivit.so"
    check "Headers exist" test -d "$sdk_dir/include/ivit"
    check "CMake config exists" test -d "$sdk_dir/lib/cmake/ivit"
    check "setup_env.sh exists" test -f "$sdk_dir/bin/setup_env.sh"
    check "INSTALL.md exists" test -f "$sdk_dir/INSTALL.md"

    # 測試 Python import (若有 wheel)
    if ls "$sdk_dir"/python/*.whl >/dev/null 2>&1; then
        check "Python wheel exists" test -f "$sdk_dir"/python/*.whl
    fi
}

verify_deb() {
    local deb="$1"
    echo "=== Verifying .deb: $deb ==="

    check ".deb file valid" dpkg-deb --info "$deb"
    check "Contains libivit.so" dpkg-deb -c "$deb" | grep -q "libivit.so"
    check "Contains headers" dpkg-deb -c "$deb" | grep -q "include/ivit"
    check "Has dependencies" dpkg-deb --info "$deb" | grep -q "Depends"
}

verify_docker() {
    local image="$1"
    echo "=== Verifying Docker image: $image ==="

    check "Image exists" docker image inspect "$image"
    check "libivit.so in image" docker run --rm "$image" test -f /usr/lib/libivit.so
    check "Python import" docker run --rm "$image" python3 -c "import ivit"
    check "Examples present" docker run --rm "$image" test -d /opt/ivit-sdk/examples
}

case "$TYPE" in
    tarball) verify_tarball "$TARGET" ;;
    deb)     verify_deb "$TARGET" ;;
    docker)  verify_docker "$TARGET" ;;
    *)       echo "Unknown type: $TYPE"; exit 1 ;;
esac

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[[ $FAIL -eq 0 ]] || exit 1
