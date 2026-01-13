# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

本專案採用 AI 虛擬團隊協作開發模式，透過多角色分工提升開發效率與品質。

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
.claude/skills/          # 團隊角色技能定義
docs/
├── PRD/                 # 產品需求文件
├── architecture/        # 架構文件 (ADR 格式)
├── design/              # 設計文件
└── test-plans/          # 測試計劃
src/                     # 原始碼
tests/                   # 測試程式碼
```

## 開發指令

```bash
# 建置
[待填入]

# 測試
[待填入]

# 執行單一測試
[待填入]

# Lint
[待填入]
```

## 專案資訊

- **專案名稱**：[待填入]
- **專案描述**：[待填入]
- **技術堆疊**：[待填入]

## 開發規範

### 程式碼
- 主要語言：[待填入]
- 風格指南：[待填入]
- 測試覆蓋率目標：80%+

### Git
- 分支策略：Git Flow / GitHub Flow
- Commit 格式：Conventional Commits
- Code Review：必要

### 文件
- 需求文件：PRD 格式
- 架構文件：ADR (Architecture Decision Records)
- API 文件：OpenAPI / Swagger
