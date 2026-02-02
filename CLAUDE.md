# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

iVIT-SDK (Innodisk Vision Intelligence Toolkit) 是宜鼎國際為其 AI 運算平台開發的統一電腦視覺推論與訓練 SDK。本 SDK 提供跨硬體平台的統一 API 介面，目前支援 Intel 和 NVIDIA 硬體，並採用可擴展架構以便未來新增更多加速器平台。

> **iVIT 命名規則**：開頭的 "i" 代表 Innodisk，VIT 代表 Vision Intelligence Toolkit。

本專案採用 AI 虛擬團隊協作開發模式，透過多角色分工提升開發效率與品質。

## 核心功能

- **統一推論 API**：一次開發，多平台部署
- **電腦視覺任務**：分類、物件偵測、語意分割、姿態估計、人臉分析
- **遷移式學習**：支援模型微調訓練
- **多後端支援**：OpenVINO (Intel)、TensorRT (NVIDIA)，可擴展架構支援更多平台
- **雙語 API**：Python 和 C++ 功能對等

## 虛擬開發團隊

| 角色 | 專長 | 技能檔案 |
|------|------|---------|
| AI 硬體產品經理 | GPU/NPU/ASIC、AI 晶片選型 | `.claude/skills/ai-hardware-product-manager/SKILL.md` |
| AI 軟體產品經理 | LLM 應用、AI 框架、產品策略 | `.claude/skills/ai-software-product-manager/SKILL.md` |
| 系統架構師 | 系統設計、技術選型、架構文件 | `.claude/skills/system-architect/SKILL.md` |
| 軟體工程師 | 程式開發、程式碼品質、測試 | `.claude/skills/software-engineer/SKILL.md` |
| QA 工程師 | 測試規劃、自動化測試、品質保證 | `.claude/skills/qa-engineer/SKILL.md` |
| UX 設計師 | 使用者體驗、介面設計、原型 | `.claude/skills/ux-designer/SKILL.md` |

## 專案結構

```
ivit-sdk/
├── CMakeLists.txt             # 頂層 CMake 設定
├── pyproject.toml             # Python 專案設定
├── include/ivit/              # C++ 公開標頭
├── src/                       # C++ 實作
│   ├── core/                  # 核心引擎
│   ├── runtime/               # 後端適配器 (openvino_runtime.cpp, tensorrt_runtime.cpp 等)
│   └── vision/                # 視覺任務
├── python/ivit/               # Python 綁定
├── models/                    # Model Zoo
├── tests/                     # 測試程式碼
├── examples/                  # 範例程式
├── docs/
│   ├── development/           # 產品需求文件 (prd.md)
│   ├── architecture/          # 架構文件 (ADR 格式)
│   ├── api/                   # API 規格文件
│   ├── deployment/            # 部署指南
│   └── tutorials/             # 教學文件
└── .claude/skills/            # 團隊角色技能定義
```

## 開發指令

```bash
# 建置 C++ 核心
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 建置 Python 套件
pip install -e .

# 測試
pytest tests/integration/
ctest --test-dir build

# 執行單一測試
pytest tests/integration/test_user_scenarios.py -v

# Lint
flake8 python/
clang-format -i src/**/*.cpp include/**/*.hpp
```

## 專案資訊

- **專案名稱**：iVIT-SDK (Innodisk Vision Intelligence Toolkit)
- **專案描述**：跨硬體平台的統一電腦視覺推論與訓練 SDK
- **技術堆疊**：
  - 語言：C++ 17、Python 3.9+
  - 綁定：pybind11
  - 後端：OpenVINO、TensorRT（可擴展架構）
  - 建置：CMake、scikit-build
  - 測試：pytest、Google Test

## 硬體支援

| 廠商 | 硬體類型 | 推論引擎 | x86 | ARM |
|------|----------|----------|-----|-----|
| Intel | CPU/iGPU/NPU/VPU | OpenVINO | ✅ | ✅ |
| NVIDIA | dGPU/Jetson | TensorRT | ✅ | ✅ |

> **可擴展架構**：SDK 設計支援動態新增硬體平台，詳見 `docs/hardware-extension.md`。

## 支援的模型任務

| 任務 | 優先級 | 支援模型 |
|------|--------|----------|
| 圖像分類 | P0 | ResNet, EfficientNet, MobileNet |
| 物件偵測 | P0 | YOLOv5/v8, SSD, Faster R-CNN |
| 語意分割 | P0 | DeepLabV3, UNet, SegFormer |
| 實例分割 | P1 | Mask R-CNN, YOLOv8-seg |
| 姿態估計 | P1 | YOLOv8-pose, HRNet |
| 人臉分析 | P1 | RetinaFace, ArcFace |

## 開發規範

### 程式碼
- 主要語言：C++ 17、Python 3.9+
- C++ 風格：Google C++ Style Guide
- Python 風格：PEP 8、Black formatter
- 測試覆蓋率目標：80%+

### Git
- 分支策略：GitHub Flow
- Commit 格式：Conventional Commits
- Code Review：必要

### 文件
- 需求文件：PRD 格式 (`docs/development/prd.md`)
- 架構文件：ADR (Architecture Decision Records) (`docs/architecture/`)
- API 文件：`docs/api/`

## 相關文件

- [PRD-001: iVIT-SDK 產品需求文件](docs/development/prd.md)
- [ADR-001: 系統架構設計](docs/architecture/adr-001-system.md)
- [API-SPEC-001: API 規格文件](docs/api/api-spec.md)
