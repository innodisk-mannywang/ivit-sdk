# 訓練模組待辦事項

> 最後更新：2026-02-04

## 已完成項目 ✅

- [x] 建立 `tests/conftest.py` 共用 fixtures
- [x] 建立 `tests/integration/test_training_workflow.py` (33 個測試案例)
- [x] 建立 `.github/workflows/test.yml` CI 測試流程
- [x] 修改 `.github/workflows/release.yml` 加入測試依賴
- [x] 建立 `docs/tutorials/training-guide.md` 訓練教學
- [x] 更新 `README.md` 新增訓練快速開始
- [x] 更新 `docs/getting-started.md` 新增訓練章節與 CUDA FAQ

---

## 待驗證項目 🔍

### 1. 執行測試套件

```bash
# 安裝測試依賴
pip install -e ".[dev,train]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 執行訓練測試
pytest tests/integration/test_training_workflow.py -v

# 執行所有測試（排除 GPU）
pytest tests/ -v -k "not gpu" --ignore=tests/cpp/

# 執行 C++ API 測試
pytest tests/integration/test_train_cpp.py -v
```

### 2. 驗證 CI Workflow

```bash
# 檢查 workflow 語法（需要 gh CLI）
gh workflow view test.yml

# 或手動 push 到 feature branch 觸發 CI
git push origin feature/train-cpp-refactor
```

### 3. 本地 GPU 測試（如有 CUDA）

```bash
# 執行 GPU 測試
pytest tests/integration/test_training_workflow.py -v -m gpu
```

---

## 後續優化項目 📋

### 短期（下一輪迭代）

- [ ] **測試覆蓋率報告**：在 CI 中加入 coverage 報告
  ```yaml
  # 在 test.yml 中加入
  - name: Run tests with coverage
    run: pytest tests/ --cov=python/ivit --cov-report=xml
  ```

- [ ] **混合精度訓練**：支援 AMP (Automatic Mixed Precision)
  ```python
  trainer = Trainer(..., use_amp=True)
  ```

- [ ] **梯度累積**：解決大 batch size 記憶體不足問題
  ```python
  trainer = Trainer(..., gradient_accumulation_steps=4)
  ```

### 中期

- [ ] **WeightedRandomSampler**：處理不平衡資料集
- [ ] **多 GPU 訓練**：支援 DataParallel / DistributedDataParallel
- [ ] **學習率 Finder**：自動找最佳學習率
- [ ] **模型剪枝**：訓練後模型壓縮

### 長期

- [ ] **物件偵測訓練**：支援 YOLO/SSD 等偵測模型訓練
- [ ] **語意分割訓練**：支援 DeepLab/UNet 等分割模型訓練
- [ ] **自監督學習**：支援 SimCLR/MoCo 等預訓練方法

---

## 已知問題 ⚠️

### CUDA 版本衝突

**問題**：PyTorch CUDA 版本與系統 CUDA 版本不一致導致 `nvjitlink` 錯誤

**解決方案**：已在文件中記錄，見 `docs/getting-started.md#常見問題`

### CI 環境限制

- GitHub Actions ubuntu-latest 無 GPU，所有 GPU 測試會被跳過
- 若需要 GPU CI 測試，需要設定 self-hosted runner

---

## 相關文件

- [訓練教學](../tutorials/training-guide.md)
- [API 規格](../api/api-spec.md)
- [快速入門](../getting-started.md)
