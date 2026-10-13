# Runtime CLI Capability Probe v0.4 — 官方整合式 CLI 優先

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
- 現階段不在版本輸出中猜測 Build Commit，因此 `identity.engine_build_commit` 為 `null`。這與 `identity.registry_upstream_commit`（文件研究 SHA）刻意分離。
- `fit-params` 尚未建立逐參數 Registry，故它只會顯示觀察到的 Help 參數，不產生偽造的「缺失參數」清單。`--include-devices` 對 `fit-params` 會拒絕執行。

## 子命令目錄觀察（延續 Runtime Capability 研究）

`llama help all` 才會顯示部分預設隱藏的子命令，例如 `bench`、`fit-params`。Runtime Probe 現在對整合式 CLI 自動讀取該目錄，將純字串列表放在 `command_results.catalog.commands_observed_in_help_all`。

如果 CLI 回傳不符合預期的格式、或目錄查詢失敗，就記錄 `null`（未能確認），**不推論該功能不存在**。對舊式獨立執行檔則標記 `not_applicable`。

## 快照

- `entrypoint.kind` 記錄 `integrated`／`legacy`，`entrypoint.command` 記錄 `serve`／`bench`。
- `program` 在 `serve`／`bench` 分別記錄 Registry 的邏輯名稱 `llama-server`／`llama-bench`；`fit-params` 沒有固定 Registry，為 `null`。不代表有獨立 exe。
- 記錄執行檔 SHA-256、版本輸出、CLI 說明中列出的旗標，以及對照固定版本 Registry 的差異。
- 說明指令失敗或沒有任何可辨識旗標時，對比欄位為 `null`（未知），不能誤判為全面不支援。
- 摘要內容有限長度、但仍可能包含路徑或環境字串；分享前應檢視或遮蔽敏感資訊。

## 尚未驗證

目前僅在 GitHub CI 裡使用合成執行檔及 mock subprocess 測試；尚未用真正的 llama.app Windows 版本、GPU 驅動或模型驗證。支援 `fit-params --help` 不代表能推導合適的 Fit 參數。旗標出現在 `--help` 中不代表該 Backend 支援、參數值有效或能在模型上提升效能。
