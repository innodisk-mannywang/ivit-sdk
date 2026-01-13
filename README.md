# Claude Project Starter

> Claude Code 專案範本，內建 AI 虛擬團隊協作開發模式，透過多角色分工提升開發效率與品質。

## 使用此範本

### 方式一：使用腳本（推薦）

```bash
# 下載並執行建立腳本
curl -fsSL https://raw.githubusercontent.com/innodisk-mannywang/claude-project-starter/main/create-project.sh | bash -s -- my-project

# 或指定目標目錄
curl -fsSL https://raw.githubusercontent.com/innodisk-mannywang/claude-project-starter/main/create-project.sh | bash -s -- my-project /path/to/workspace
```

### 方式二：手動操作

```bash
# Clone 範本
git clone https://github.com/innodisk-mannywang/claude-project-starter.git my-project

# 進入專案目錄並重新初始化
cd my-project
rm -rf .git
git init

# 編輯 CLAUDE.md 填入專案資訊
# 編輯 README.md 更新專案說明
```

## 虛擬開發團隊

本專案配備完整的 AI 開發團隊，每個角色都有專精的技能和職責：

| 角色 | 專長 | 適用場景 |
|------|------|---------|
| **AI 硬體產品經理** | GPU/NPU/ASIC、AI 晶片選型 | AI 硬體規格、供應商評估 |
| **AI 軟體產品經理** | LLM 應用、AI 框架、產品策略 | AI 功能設計、框架選型 |
| **系統架構師** | 系統設計、技術選型、架構文件 | 架構設計、技術決策 |
| **軟體工程師** | 程式開發、程式碼品質、測試 | 功能實作、程式碼審查 |
| **QA 工程師** | 測試規劃、自動化測試、品質保證 | 測試案例、錯誤報告 |
| **UX 設計師** | 使用者體驗、介面設計、原型 | UI/UX 設計、流程設計 |

## 角色使用指南

### 呼叫特定角色

當需要特定角色協助時，請使用以下方式：

```
「以 [角色名稱] 的身份，[任務描述]」
「請 [角色名稱] 來 [任務描述]」
「從 [角色名稱] 的角度，[問題或需求]」
```

### 使用範例

#### AI 硬體產品經理
```
「以 AI 硬體產品經理的身份，評估 NVIDIA H100 vs AMD MI300X 用於 LLM 訓練」
「請 AI 硬體 PM 推薦邊緣 AI 裝置的晶片方案」
```

#### AI 軟體產品經理
```
「以 AI 軟體產品經理的身份，設計 RAG 系統的產品規格」
「請 AI 軟體 PM 評估 LangChain vs LlamaIndex」
```

#### 系統架構師
```
「以系統架構師的身份，設計這個系統的整體架構」
「請架構師評估微服務 vs 單體架構的優缺點」
```

#### 軟體工程師
```
「以軟體工程師的身份，實作這個 API 端點」
「請工程師審查這段程式碼」
```

#### QA 工程師
```
「以 QA 工程師的身份，撰寫這個功能的測試案例」
「請 QA 設計自動化測試策略」
```

#### UX 設計師
```
「以 UX 設計師的身份，設計這個功能的使用者流程」
「請 UX 設計師建立使用者人物誌」
```

## 團隊協作模式

### 模式一：單一角色

適用於單一專業領域的任務：

```
「請軟體工程師實作用戶登入功能」
```

### 模式二：角色接力

適用於需要多專業協作的任務：

```
「這個新功能請按照以下順序處理：
1. 產品經理定義需求和使用者故事
2. 系統架構師設計技術方案
3. 軟體工程師實作功能
4. QA 工程師撰寫測試案例」
```

### 模式三：角色會議

適用於需要多角度評估的決策：

```
「請召開團隊會議，討論這個技術選型：
- 架構師評估技術可行性
- 工程師評估實作難度
- PM 評估產品需求符合度
- QA 評估可測試性」
```

### 模式四：完整開發流程

適用於從零開始的專案或功能：

```
「請團隊協作開發 [功能名稱]：

Phase 1 - 需求分析
→ AI 軟體 PM：定義功能需求和驗收標準

Phase 2 - 設計
→ 系統架構師：設計技術架構
→ UX 設計師：設計使用者流程和介面

Phase 3 - 實作
→ 軟體工程師：實作功能程式碼

Phase 4 - 測試
→ QA 工程師：執行測試和驗證

Phase 5 - 審查
→ 全團隊：Code Review 和最終確認」
```

## 專案結構

```
.
├── .claude/
│   └── skills/                  # 團隊角色技能定義
│       ├── ai-hardware-product-manager/
│       ├── ai-software-product-manager/
│       ├── system-architect/
│       ├── software-engineer/
│       ├── qa-engineer/
│       └── ux-designer/
├── docs/                        # 文件
│   ├── PRD/                     # 產品需求文件
│   ├── architecture/            # 架構文件
│   ├── design/                  # 設計文件
│   └── test-plans/              # 測試計劃
├── src/                         # 原始碼
├── tests/                       # 測試程式碼
├── CLAUDE.md                    # Claude Code 技術指引
└── README.md                    # 本檔案
```

## 快速開始

### 開始新功能開發

```
「我想開發 [功能描述]，請團隊協作：
1. PM 先定義需求
2. 架構師設計方案
3. 工程師評估工時
4. QA 定義測試策略」
```

### 技術問題諮詢

```
「我遇到 [問題描述]，請 [最相關的角色] 提供建議」
```

### 程式碼審查

```
「請軟體工程師審查以下程式碼：
[貼上程式碼]」
```

### 架構決策

```
「我們需要決定 [技術選項 A] vs [技術選項 B]，
請架構師和工程師一起評估優缺點」
```

## 備註

- Claude 會根據指示自動切換角色
- 每個角色會依據其 SKILL.md 中的專業知識回應
- 可以在對話中隨時切換角色
- 複雜任務建議使用角色接力或團隊會議模式
