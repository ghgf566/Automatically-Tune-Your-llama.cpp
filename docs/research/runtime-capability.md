# Runtime CLI Capability Probe v0.6 — 官方整合式 CLI 優先

ATYL 目前只做 **唯讀探測**。不啟動伺服器、不載入模型、不做 Bench、不安裝或更新引擎。

## 官方 Windows 安裝版（推薦）

請先由使用者透過 [官方安裝通道](official-installation.md) 完成安裝，在 PowerShell 的 ATYL 儲存庫根目錄執行：

```powershell
python tools/probe_runtime.py --command serve --output my-server-capability.json
python tools/probe_runtime.py --command bench --output my-bench-capability.json
python tools/probe_runtime.py --command fit-params --output my-fit-capability.json
```

`--binary` 可省略，預設透過 PATH 尋找 `llama.exe`。也可以明確指定：

```powershell
python tools/probe_runtime.py --binary "$env:LOCALAPPDATA\Microsoft\WindowsApps\llama.exe" --command serve
```

相容舊式獨立執行檔，但**只在明確指定路徑時**使用：

```powershell
python tools/probe_runtime.py --binary "D:\old-llama\llama-server.exe" --command serve
python tools/probe_runtime.py --binary "D:\old-llama\llama-bench.exe" --command bench
```

## 會呼叫什麼？

- 整合式 `serve`：`llama version`、`llama help all`、`llama serve --help`；可選 `llama serve --list-devices`。
- 整合式 `bench`：`llama version`、`llama help all`、`llama bench --help`；可選 `llama bench --list-devices`。
- 整合式 `fit-params`：`llama version`、`llama help all`、`llama fit-params --help`；**不**執行模型 Fitting。
- 舊式執行檔：`llama-server --version`、`llama-server --help`（或 `llama-bench` 的對應指令）。

必須傳入 `--include-devices` 才會執行裝置查詢（可能初始化 GPU 驅動）。每個命令都有 timeout。透過 `subprocess` argv 陣列執行，**不使用 shell**，沒有任何自動下載或更新。

## 集中路由與身份識別

- `tools/engine_adapter.py` 是單一指令分派表，支援 `serve`、`bench` 以及唯讀 `fit-params --help`。過去探測器和驗證器各寫一份路由規則的情況已消除。
- `tools/engine_identity.py` 提供獨立的 SHA-256 版本識別與候選能力快取 Key：輸入包含執行檔 SHA-256、子命令、固定 Registry 上游 Commit 和實際版本輸出。
- 只有 `llama version` 成功且 `--help` 能辨識參數時，身份資料才設 `cacheable: true`。**目前尚未儲存或重用能力快取**；只生成可比較的 Key。
- 對已核對的官方 `version: ... (build N, commit HEX)` 單一格式，`identity.engine_build_commit` 可記錄**執行檔回報的短 Commit**，並分別保留 `engine_build_number`、`engine_build_commit_form`。格式未知或模糊時仍為 `null`，絕不複製 Registry 文件 SHA。
- `fit-params` 尚無完整參數 Registry；現在有 13 個來源固定的**精選參數能力基準** `data/fit-capability-reference.json`，只比對 Help 出現與否，不能當成 197 項完整清單，也不會聲稱模型可 Fit。`--include-devices` 對 `fit-params` 會拒絕執行。

## 子命令目錄觀察（延續 Runtime Capability 研究）

`llama help all` 才會顯示部分預設隱藏的子命令，例如 `bench`、`fit-params`。Runtime Probe 現在對整合式 CLI 自動讀取該目錄，將純字串列表放在 `command_results.catalog.commands_observed_in_help_all`。

如果 CLI 回傳不符合預期的格式、或目錄查詢失敗，就記錄 `null`（未能確認），**不推論該功能不存在**。對舊式獨立執行檔則標記 `not_applicable`。

## 裝置資訊（需選擇性啟用）

新增 `tools/device_inventory.py`，根據固定 upstream `common/arg.cpp` 中 `common_print_available_devices()` 的格式，辨識非 CPU 裝置 ID、名稱、記憶體總量與可用量。必須明確給 `--include-devices`，而且 `--help` 已觀察到 `--list-devices` 才會執行。

`device_inventory` 只有三種狀態：`observed`、`none`、`unknown`。其中未知會保留原因，且不返回空裝置陣列以免誤判；空陣列只在成功解析 `(none)` 時產生。結構由 [JSON Schema](../../schemas/device-inventory.schema.json) 驗證。

這個輸出**排除 CPU**、沒有 NPU 認證，也不提供模型算子／Kernel 相容性或 Benchmark 效能保證。

實機檢查請依照 [Windows 官方版唯讀 Smoke Test](windows-smoke-test.md) 操作。

## 快照

- `entrypoint.kind` 記錄 `integrated`／`legacy`，`entrypoint.command` 記錄 `serve`／`bench`。
- `program` 在 `serve`／`bench` 分別記錄 Registry 的邏輯名稱 `llama-server`／`llama-bench`；`fit-params` 沒有固定 Registry，為 `null`。不代表有獨立 exe。
- 記錄執行檔 SHA-256、版本輸出、CLI 說明中列出的旗標，以及對照固定版本 Registry 的差異。
- 說明指令失敗或沒有任何可辨識旗標時，對比欄位為 `null`（未知），不能誤判為全面不支援。
- 摘要內容有限長度、但仍可能包含路徑或環境字串；分享前應檢視或遮蔽敏感資訊。

## 依 Windows 實機資料改進

原始三份實測 JSON 沒有直接存入儲存庫，只保留去識別化的最小回歸樣本。診斷欄位 `diagnostics` 分辨「文件已記載但 Help 未觀察到」以及「Help 有但固定文件未登錄」；對 Bench `--rpc` 會標記上游 `llama_supports_rpc()` 條件式顯示，**不是斷言不支援 RPC**。尾端只有 `-` 的不完整旗標會由解析器過濾。

Fit Help 透過 `fit_reference` 對照 13 筆精選旗標，特別注意 `--fit-print on` 是估算記憶體輸出，並非已調整好的啟動參數。詳見 [實機回歸證據與 Fit 參數](windows-build11574-fit-reference.md)。

## 尚未驗證

目前已收到並分析一組使用者自行執行的官方 llama.app Windows 版本輸出；GitHub CI 仍只使用去識別化摘要與 mock subprocess，**CI 沒有在真實 Windows GPU 上執行**。支援 `fit-params --help` 不代表能推導合適的 Fit 參數。旗標出現在 `--help` 中不代表該 Backend 支援、參數值有效或能在模型上提升效能。
