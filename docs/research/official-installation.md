# ATYL 引擎來源：官方 llama.app Windows 通道

## 決定

ATYL 以 **已由使用者安裝、並在 PATH 可找到的官方整合式 `llama.exe`** 為預設。ATYL 不包含 llama.cpp 推理引擎本體、不打包平台專屬 DLL/ZIP，不接手 CUDA/ROCm/Vulkan/CPU 編譯版本選擇，也不執行官方安裝與更新腳本。

在 Windows **PowerShell** 中，官方目前提供：

```powershell
irm https://llama.app/install.ps1 | iex
Get-Command llama
llama version
llama help all
```

安裝腳本 [來源](https://llama.app/install.ps1)（動態變更，2026-10-14 檢視）顯示：

- 依序探測 CUDA、ROCm、Vulkan、CPU，選取其預編譯整合式版本；CUDA 可能需要 Toolkit、ROCm 可能需要 HIP SDK，並不保證最終使用 GPU 後端。
- 暫存下載目錄位於 LocalAppData 中；通過版本驗證後，將實際 `llama.exe` 移到 `%LOCALAPPDATA%\Microsoft\WindowsApps\llama.exe`。
- 官方透過使用者現有 PATH 讓 `llama` 可呼叫；ATYL **不依賴固定路徑**，優先用 PATH 探測，讓未來安裝路徑可變動。
- 安裝腳本可能會從 Hugging Face 官方 bucket 下載檔案；不是由 ATYL 自行解析 GitHub Releases。

## CLI 指令映射（上游 `app/llama.cpp`）

| ATYL 邏輯功能 | 官方整合式指令 | 研究資料來源 |
| --- | --- | --- |
| 版本 | `llama version` | 官方 CLI routing |
| 伺服器 | `llama serve [options]` | `llama-server` README 的旗標 |
| 效能測試 | `llama bench [options]` | `llama-bench` README 的旗標 |
| 記憶體配置 | `llama fit-params [options]` | 尚未納入當前 Registry |
| 其他可用命令 | `llama help all` | 官方 CLI routing |
| 官方更新 | `llama update` | 僅官方安裝版支援，ATYL 不自動呼叫 |

以上命令經 [上游固定 commit 的 `app/llama.cpp`](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/app/llama.cpp) 核對。**不能因為同一支 exe，就把 serve 和 bench 的參數混為一談**。

## 目前實作

- `probe_runtime.py` 不帶 `--binary` 時，會透過 PATH 尋找官方 `llama`。
- 使用 `--command serve` 或 `--command bench` 呼叫相應子命令的 `--help`，對照原有 257／45 筆邏輯旗標。
- 將 SHA-256、`version` 輸出及查詢到的 CLI 內容保存為可觀測資料。可明確指定獨立 `llama-server.exe` 或 `llama-bench.exe` 作為相容模式。
- **不呼叫** `llama serve`（沒有 `--help`）、`llama bench`（沒有 `--help`）、`llama update`，也不對系統安裝、模型與驅動做修改。

## 後續開發限制

1. 每次真正執行前，確認 `binary sha256 + version + subcommand + registry revision`；官方 `llama update` 後需重新探測，使舊快取失效。
2. **後端不能單靠安裝器偵測假設**；還需要獨立的可用裝置及實際模型載入驗證。
3. 若最終做 Runner，必須傳 argv 陣列如 `['llama.exe', 'serve', '--model', '...']`，不拼接 shell 字串；並處理取消、timeout、隱私與進程清理。
4. GitHub CI 的 mock 測試不代表已在官方 Windows 安裝版、CUDA 或其他硬體上實測。
