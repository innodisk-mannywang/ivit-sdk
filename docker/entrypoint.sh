#!/bin/bash
# iVIT-SDK Docker Entrypoint
#
# 初始化環境並執行命令

set -e

# 顯示歡迎訊息
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           iVIT-SDK Verification Environment                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Test Data:                                                  ║"
echo "║    Images: /opt/test-data/images/                            ║"
echo "║    Models: /opt/test-data/models/                            ║"
echo "║                                                              ║"
echo "║  Quick Commands:                                             ║"
echo "║    verify-sdk.sh          - Run full verification            ║"
echo "║    verify-sdk.sh --help   - Show help                        ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 檢查 SDK 原始碼是否掛載
if [ -d "/workspace/ivit-sdk" ]; then
    echo "[INFO] SDK source mounted at /workspace/ivit-sdk"
    cd /workspace/ivit-sdk
else
    echo "[WARN] SDK source not mounted. Mount with:"
    echo "       -v \$(pwd):/workspace/ivit-sdk"
fi

# 執行命令
exec "$@"
