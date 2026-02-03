# iVIT-SDK 發布自動化指南

本文件說明 iVIT-SDK 的自動化打包與發布流程。

## 概述

iVIT-SDK 提供三種交付方案，支援三個目標平台：

| 平台 ID | 架構 | 後端 | 適用場景 |
|---------|------|------|----------|
| `x86_64-nvidia` | x86_64 | TensorRT | 桌機/伺服器 + NVIDIA GPU |
| `x86_64-intel` | x86_64 | OpenVINO | 桌機/伺服器 + Intel CPU/iGPU/NPU |
| `aarch64-jetson` | aarch64 | TensorRT | NVIDIA Jetson 嵌入式平台 |

| 交付方案 | 說明 | 依賴處理 |
|----------|------|----------|
| **Tarball** | 預編譯壓縮包 | 內含所有依賴（self-contained） |
| **Deb** | Debian 套件 | 依賴系統套件（透過 apt） |
| **Docker** | Container image | 內含完整執行環境 |

---

## 自動化流程

### 觸發方式

1. **推送 Git Tag**（推薦用於正式發布）
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **手動觸發**（適合測試或 hotfix）
   - 前往 GitHub Actions → Release workflow
   - 點擊「Run workflow」
   - 填入版本號與目標平台

### 流程圖

```
┌─────────────┐
│  Push Tag   │  或  手動觸發
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Prepare   │  解析版本、建立平台矩陣
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Tarball x3  │ │   Deb x3    │ │  Docker x3  │  平行建置
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │              │              │
       └──────────────┼──────────────┘
                      ▼
              ┌─────────────┐
              │   Publish   │  上傳至 GitHub Release
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │   Notify    │  發送通知
              └─────────────┘
```

---

## 手動建置

如需在本機手動建置，可使用以下腳本：

### 一鍵建置所有方案

```bash
# 建置單一平台的所有方案
./scripts/release.sh x86_64-nvidia

# 建置所有平台
./scripts/release.sh all

# 跳過特定方案
./scripts/release.sh x86_64-intel --skip-docker --skip-deb
```

### 個別建置

```bash
# Tarball
./scripts/build_tarball.sh x86_64-nvidia
# 產出: build-tarball-x86_64-nvidia/ivit-sdk-1.0.0-x86_64-nvidia.tar.gz

# Deb
./scripts/build_deb.sh x86_64-intel
# 產出: build-deb-x86_64-intel/ivit-sdk_1.0.0_amd64-intel.deb

# Docker
./scripts/build_docker.sh aarch64-jetson
# 產出: innodisk/ivit-sdk:1.0.0-jetson
```

### 驗證產出

```bash
# 驗證 tarball
./scripts/verify_package.sh tarball build-tarball-x86_64-nvidia/ivit-sdk-1.0.0-x86_64-nvidia.tar.gz

# 驗證 deb
./scripts/verify_package.sh deb build-deb-x86_64-intel/ivit-sdk_1.0.0_amd64-intel.deb

# 驗證 docker
./scripts/verify_package.sh docker innodisk/ivit-sdk:1.0.0-nvidia
```

---

## 產出物說明

### Tarball 結構

```
ivit-sdk-1.0.0-x86_64-nvidia/
├── lib/
│   ├── libivit.so              # 核心函式庫
│   └── cmake/ivit/             # CMake 設定檔
├── include/ivit/               # C++ 標頭檔
├── deps/lib/                   # 打包的依賴庫 (TensorRT/OpenVINO)
├── python/
│   └── ivit_sdk-1.0.0-*.whl    # Python wheel
├── bin/
│   └── setup_env.sh            # 環境設定腳本
├── examples/                   # 範例程式
└── INSTALL.md                  # 安裝說明
```

### Deb 套件

- 安裝路徑：`/usr/lib`、`/usr/include`
- 自動宣告依賴：`libopencv-dev`、`openvino` 或 `libnvinfer-dev`
- 檔名格式：`ivit-sdk_<version>_<arch>-<backend>.deb`

### Docker Image

- Registry：`ghcr.io/<org>/ivit-sdk`
- Tag 格式：`<version>-<backend>` (e.g., `1.0.0-nvidia`)
- 包含：預編譯 SDK + Python wheel + 範例程式
- 不含：原始碼

---

## CI/CD 設定

### 必要的 Secrets

| Secret | 說明 |
|--------|------|
| `GITHUB_TOKEN` | 自動提供，用於推送 Docker image 到 GHCR |

### Self-hosted Runner（aarch64）

Jetson 平台需要 ARM64 runner：

1. 在 Jetson 設備上安裝 GitHub Actions Runner
2. 加入 label：`self-hosted-arm64`
3. 確保已安裝：CUDA、TensorRT、OpenCV、CMake

### 自訂 Docker Registry

修改 `.github/workflows/release.yml` 中的環境變數：

```yaml
env:
  REGISTRY: your-registry.com
  IMAGE_NAME: your-org/ivit-sdk
```

---

## 版本管理

### 版本號規則

遵循 [Semantic Versioning](https://semver.org/)：

- `MAJOR.MINOR.PATCH` (e.g., `1.0.0`)
- 預發布版：`1.0.0-rc1`、`1.0.0-beta`

### 發布檢查清單

1. [ ] 更新 `CMakeLists.txt` 中的版本號
2. [ ] 更新 `pyproject.toml` 中的版本號
3. [ ] 更新 CHANGELOG
4. [ ] 確保所有測試通過
5. [ ] 建立並推送 tag

```bash
# 更新版本後
git add CMakeLists.txt pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 1.1.0"
git tag v1.1.0
git push origin main --tags
```

---

## 故障排除

### Tarball 建置失敗

**症狀**：CMake 找不到 OpenVINO/TensorRT

**解決**：
```bash
# Intel 平台
source /opt/intel/openvino/setupvars.sh

# NVIDIA 平台
export TENSORRT_ROOT=/usr/local/tensorrt
```

### Deb 安裝後找不到 library

**症狀**：`error while loading shared libraries: libivit.so`

**解決**：
```bash
sudo ldconfig
```

### Docker 建置失敗

**症狀**：Multi-stage build 中找不到檔案

**檢查**：
- 確認 `.dockerignore` 沒有排除必要檔案
- 確認 COPY 指令路徑正確

---

## 相關檔案

| 檔案 | 用途 |
|------|------|
| `.github/workflows/release.yml` | CI/CD 工作流程定義 |
| `scripts/build_tarball.sh` | Tarball 建置腳本 |
| `scripts/build_deb.sh` | Deb 建置腳本 |
| `scripts/build_docker.sh` | Docker 建置腳本 |
| `scripts/release.sh` | 整合建置腳本 |
| `scripts/verify_package.sh` | 驗證腳本 |
| `packaging/INSTALL.md` | 客戶安裝說明 |
| `docker/Dockerfile.release-*` | 各平台 Dockerfile |
