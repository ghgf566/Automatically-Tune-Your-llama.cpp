# Runtime CLI Capability Probe v0.2 — 官方整合式 CLI 優先

ATYL 目前只做 **唯讀探測**。不啟動伺服器、不載入模型、不做 Bench、不安裝或更新引擎。

## 官方 Windows 安裝版（推薦）

請先由使用者透過 [官方安裝通道](official-installation.md) 完成安裝，在 PowerShell 的 ATYL 儲存庫根目錄執行：

```powershell
python tools/probe_runtime.py --command serve --output my-server-capability.json
python tools/probe_runtime.py --command bench --output my-bench-capability.json
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

- 整合式 `serve`：`llama version`、`llama serve --help`；可選 `llama serve --list-devices`。
- 整合式 `bench`：`llama version`、`llama bench --help`；可選 `llama bench --list-devices`。
- 舊式執行檔：`llama-server --version`、`llama-server --help`（或 `llama-bench` 的對應指令）。

必須傳入 `--include-devices` 才會執行裝置查詢（可能初始化 GPU 驅動）。每個命令都有 timeout。透過 `subprocess` argv 陣列執行，**不使用 shell**，沒有任何自動下載或更新。

## 快照

- `entrypoint.kind` 記錄 `integrated`／`legacy`，`entrypoint.command` 記錄 `serve`／`bench`。
- `program` 仍然使用 Registry 的邏輯名稱 `llama-server`／`llama-bench`，不代表必須存在這些獨立 exe。
- 記錄執行檔 SHA-256、版本輸出、CLI 說明中列出的旗標，以及對照固定版本 Registry 的差異。
- 說明指令失敗或沒有任何可辨識旗標時，對比欄位為 `null`（未知），不能誤判為全面不支援。
- 摘要內容有限長度、但仍可能包含路徑或環境字串；分享前應檢視或遮蔽敏感資訊。

## 尚未驗證

目前僅在 GitHub CI 裡使用合成執行檔及 mock subprocess 測試；尚未用真正的 llama.app Windows 版本、GPU 驅動或模型驗證。旗標出現在 `--help` 中不代表該 Backend 支援、參數值有效或能在模型上提升效能。
