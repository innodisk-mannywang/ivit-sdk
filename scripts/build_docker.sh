#!/bin/bash
# build_docker.sh — 建置 + tag Docker release image
#
# Usage: ./scripts/build_docker.sh <platform> [--push]
#   platform: x86_64-nvidia | x86_64-intel | aarch64-jetson

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION=$(grep 'project(ivit VERSION' "$PROJECT_ROOT/CMakeLists.txt" | sed 's/.*VERSION \([^ )]*\).*/\1/')

PLATFORM="${1:-}"
PUSH="${2:-}"

if [[ -z "$PLATFORM" ]]; then
    echo "Usage: $0 <platform> [--push]"
    echo "  Platforms: x86_64-nvidia | x86_64-intel | aarch64-jetson"
    exit 1
fi

REGISTRY="${IVIT_DOCKER_REGISTRY:-innodisk}"

case "$PLATFORM" in
    x86_64-nvidia)
        DOCKERFILE="docker/Dockerfile.release-nvidia"
        TAG="${REGISTRY}/ivit-sdk:${VERSION}-nvidia"
        ;;
    x86_64-intel)
        DOCKERFILE="docker/Dockerfile.release-intel"
        TAG="${REGISTRY}/ivit-sdk:${VERSION}-intel"
        ;;
    aarch64-jetson)
        DOCKERFILE="docker/Dockerfile.release-jetson"
        TAG="${REGISTRY}/ivit-sdk:${VERSION}-jetson"
        ;;
    *)
        echo "Error: Unknown platform '$PLATFORM'"
        exit 1
        ;;
esac

echo "=== Building Docker image for $PLATFORM ==="
echo "  Dockerfile: $DOCKERFILE"
echo "  Tag:        $TAG"

docker build \
    -f "$PROJECT_ROOT/$DOCKERFILE" \
    -t "$TAG" \
    -t "${TAG%-*}:latest-${PLATFORM##*-}" \
    "$PROJECT_ROOT"

echo ""
echo "=== Image built: $TAG ==="

if [[ "$PUSH" == "--push" ]]; then
    echo "Pushing $TAG ..."
    docker push "$TAG"
fi
