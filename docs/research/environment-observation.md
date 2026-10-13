# 日常／低背景負載環境觀察與比較 v0.1

ATYL 現在會在明確使用 `--execute` 執行 `llama fit-params` 記憶體估算時，於**執行前後**各取樣一次系統實體 RAM 和 GGML 回報的裝置記憶體餘量。這是一種 *headroom snapshot*，不是模型實際佔用量，也**不是執行期間峰值記憶體**。

## 不會做的事情

- 不自動關閉 Wallpaper Engine、瀏覽器、錄影軟體或任何背景行程。
- 不從正在執行的行程中判斷你是不是真的「低背景負載」。`daily` 與 `reduced_background` 完全是使用者宣告的實驗標籤。
- 不從 GPU 餘量下降值推定模型 VRAM 用量。估算結束後 GPU 記憶體可能已被釋放；其他程式的活動也會影響數字。
- 不把兩次環境數據當作性能 Benchmark、模型載入成功、峰值 RAM/VRAM 或最佳 Offload 的證據。

## Windows 實機操作

從 ATYL 專案根目錄執行 `git pull`，準備先前兩個 Snapshot：`fit-capability-v2.json`、`devices-capability-v2.json`，以及本機模型。示範路徑請換成你自己的。

### 1. 保留 Wallpaper Engine 的日常環境

```powershell
python tools/fit_estimate.py --model "D:\models\my-model.gguf" `
  --fit-snapshot fit-capability-v2.json `
  --device-snapshot devices-capability-v2.json `
  --device Vulkan0 --context 4096 --gpu-layers 18 `
  --scenario daily --background-note "Wallpaper Engine running" `
  --execute --output fit-daily.json
```

### 2. 手動暫停 Wallpaper Engine 等程式後重做一次

請先**自行**暫停或關閉想要移除的背景負載；ATYL 不會操作這些程式。

```powershell
python tools/fit_estimate.py --model "D:\models\my-model.gguf" `
  --fit-snapshot fit-capability-v2.json `
  --device-snapshot devices-capability-v2.json `
  --device Vulkan0 --context 4096 --gpu-layers 18 `
  --scenario reduced_background --background-note "Wallpaper Engine manually paused" `
  --execute --output fit-reduced.json
```

這裡的 4096 Context 與 18 層只是**相同候選的控制變數**，不是 ATYL 建議的最佳值。兩次測試最好保持引擎版本、模型、設備、候選參數一致；兩次環境測試可能互相影響，因此要留意溫度、GPU Driver、磁碟快取和其他背景活動。

### 3. 僅比較有觀察到的差異

```powershell
python tools/compare_scenarios.py --daily fit-daily.json `
  --reduced fit-reduced.json --output fit-environment-comparison.json
```

只有當兩份結果的執行檔 SHA-256、模型記錄（檔名與大小）與候選設定一致時才比較。出現未知數字時，差值為 `null`，**不是 0 MiB**。相同檔名大小並不構成模型內容完全相同的密碼學證明。

## 記錄格式

- `environment.scenario`：`daily`／`reduced_background`。
- `environment.background_note`：使用者自填的描述（例如 Wallpaper Engine 正在執行），不自動掃描行程。
- `environment.before.ram`／`after.ram`：Windows `GlobalMemoryStatusEx` 回報的總實體 RAM 與可用量（MiB）。Linux 研究 CI 則使用 `/proc/meminfo`，其他平台標未知。
- `environment.before.gpu`／`after.gpu`：使用相同已驗證的官方 `llama.exe`，再次呼叫 `llama serve --list-devices`，記錄所選 `Vulkan0` 等裝置的 GGML 總量／可用量；裝置無法列舉就標 `unknown`。
- `available_memory_change`：估算後可用量減估算前可用量；可為正或負。`peak_gpu_usage_mib`、`peak_ram_usage_mib` 一律為 `null`，因為沒有連續峰值採樣。
- `estimate.status`：`completed`／`nonzero_exit`／`timeout`／`process_error`，僅表示官方估算子程序結果，**不是模型實際載入測試**。

取樣本身會初始化 GGML 裝置，可能干擾瞬時空閒記憶體。即使 GPU 數字成功，仍不能直接與 Windows Task Manager 專用 VRAM 指標混用。此階段也尚未進行真實 Windows 實機驗證，CI 只有模擬系統資訊及裝置輸出。

## 下一步

拿到首份 `fit-daily.json` 後先驗證引擎估算原始輸出的格式，才新增精準的記憶體需求 Parser。模型載入成功率、運算效能及尖峰記憶體的量測，應在真正的 Feasibility Runner 階段獨立設計。
