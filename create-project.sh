#!/bin/bash

# Claude Project Starter - 新專案建立腳本
# 用法: ./create-project.sh <專案名稱> [目標目錄]

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 檢查參數
if [ -z "$1" ]; then
    echo -e "${RED}錯誤: 請提供專案名稱${NC}"
    echo ""
    echo "用法: $0 <專案名稱> [目標目錄]"
    echo ""
    echo "範例:"
    echo "  $0 my-awesome-project"
    echo "  $0 my-awesome-project /path/to/workspace"
    exit 1
fi

PROJECT_NAME="$1"
TARGET_DIR="${2:-.}"
TEMPLATE_REPO="https://github.com/innodisk-mannywang/claude-project-starter.git"
PROJECT_PATH="$TARGET_DIR/$PROJECT_NAME"

# 檢查目標是否已存在
if [ -d "$PROJECT_PATH" ]; then
    echo -e "${RED}錯誤: 目錄 '$PROJECT_PATH' 已存在${NC}"
    exit 1
fi

echo -e "${GREEN}🚀 建立新專案: $PROJECT_NAME${NC}"
echo ""

# 1. Clone 範本
echo -e "${YELLOW}[1/5] 下載範本...${NC}"
git clone --depth 1 "$TEMPLATE_REPO" "$PROJECT_PATH"

# 2. 移除原本的 git
echo -e "${YELLOW}[2/5] 初始化專案...${NC}"
rm -rf "$PROJECT_PATH/.git"

# 3. 更新 README.md 標題
echo -e "${YELLOW}[3/5] 更新專案名稱...${NC}"
sed -i "s/# Claude Project Starter/# $PROJECT_NAME/" "$PROJECT_PATH/README.md"
sed -i "s/claude-project-starter/$PROJECT_NAME/g" "$PROJECT_PATH/README.md"

# 4. 初始化新的 git repo
echo -e "${YELLOW}[4/5] 初始化 Git...${NC}"
cd "$PROJECT_PATH"
git init
git add .
git commit -m "Initial commit: $PROJECT_NAME (from claude-project-starter template)"

# 5. 完成
echo -e "${YELLOW}[5/5] 完成!${NC}"
echo ""
echo -e "${GREEN}✅ 專案已建立: $PROJECT_PATH${NC}"
echo ""
echo "下一步:"
echo "  cd $PROJECT_PATH"
echo "  # 編輯 CLAUDE.md 填入專案資訊"
echo "  # 開始開發!"
