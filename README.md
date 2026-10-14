# ATYL — Automatically Tune Your llama.cpp

ATYL 研究如何根據使用者的硬體、模型、服務負載與目標，自動找出合適的 llama.cpp 執行參數。

目前是**規格化與受控 CLI 原型**階段：已有 Runtime 探測、GGUF Metadata 檢查、單候選記憶體估算、記憶體輸出解析、環境比較、target/draft 配對預檢及明確 `--execute` 閘門的本機模型載入可行性檢查；尚無 GUI、完整 Tuner、Benchmark Runner 或可靠的 draft-aware 峰值記憶體估算。

## 官方 llama.cpp 安裝與執行入口

ATYL **優先支援官方 [llama.app](https://llama.app) 安裝的整合式 `llama.exe`**，不下載或打包 GitHub Releases 中的獨立推理引擎，也不自行替換 llama.cpp 核心。預設由 PATH 尋找 `llama`，以 `llama serve`／`llama bench` 分別對應原有的 server／bench 參數研究；既有獨立執行檔保留為明確指定時的相容模式。

在 **PowerShell** 安裝（`irm`／`iex` 不是 CMD 原生命令）：

```powershell
irm https://llama.app/install.ps1 | iex
llama version
llama help all
llama serve --help
llama bench --help
```

Windows 安裝腳本目前將實際執行檔放在 `%LOCALAPPDATA%\Microsoft\WindowsApps\llama.exe`；依可用 CUDA、ROCm、Vulkan 與 CPU 版本選擇預編譯檔。**實際選到的後端必須以本機檢查為準。** 安裝和更新由官方通道／使用者自行執行，ATYL 目前不自動執行遠端腳本或 `llama update`。

```powershell
python tools/probe_runtime.py --command serve --output server-capability.json
python tools/probe_runtime.py --command bench --output bench-capability.json
python tools/probe_runtime.py --command fit-params --output fit-capability.json
```

探測器透過統一的 Command Adapter 只呼叫版本與說明指令；上述命令不會載入模型或啟動伺服器。完整細節見 [官方安裝策略](docs/research/official-installation.md) 與 [Runtime CLI 探測文件](docs/research/runtime-capability.md)。

## 目前文件

- [文件索引](docs/README.md)
- [能力與參數空間規格](docs/specification/parameter-space.md)
- [CLI Registry v1](docs/specification/cli-registry.md)
- [Runtime CLI 探測 v0.6](docs/research/runtime-capability.md)
- [實驗 Profile／Result 契約 v1](docs/specification/experiment-contracts.md)
- [量測與實驗方法](docs/research/benchmark-methodology.md)
- [初步系統架構](docs/research/system-architecture.md)
- [上游來源與版本](docs/sources.md)

## 進入 Feasibility 階段前的整合驗收

已完成從 CLI／GGUF 探測、受控 Fit、完整 Profile、MTP Draft N 候選，到 Candidate ID／Run Record 的跨模組程式碼檢查與合成模型整合測試。GGUF Inspector 現會輸出**標頭前綴 SHA-256**，防止先前 Fit 的記錄與不相干的 Metadata JSON 僅憑同檔名／大小被連結；舊 Metadata JSON 需重新產生才能使用此身分關聯機制。具體驗收結果與未完成的實機功能見 [Feasibility 前最終驗收](docs/research/pre-feasibility-audit.md)。

## 目前的架構分層

- `tools/engine_adapter.py`：官方整合式 CLI 與 Legacy 命令路由的唯一來源。
- `tools/engine_identity.py`：實際二進位 SHA-256、版本回報與獨立的 Registry 來源 Commit；更新版本時生成不同的能力快取 Key。
- `schemas/workload-profile.schema.json`／`run-record.schema.json` 為 1.1.0；記憶體估算契約為 0.3.0，Runtime Probe 為 1.1.0。引擎 Build 與文件 Registry SHA 不互相冒充。
- `fit-params` 支援唯讀 `--help` 與**明確 `--execute` 授權**的單一候選記憶體估算。13 項精選 Fit 旗標為研究基準，並非完整參數 Registry。
- `tools/device_inventory.py`：可選擇性列舉 GGML 回報的非 CPU 裝置及可用記憶體；尚未驗證實際 Backend／模型能力。
- [Windows 手動 Smoke Test](docs/research/windows-smoke-test.md)：操作官方已安裝的 llama.exe，不需要啟動推理服務。

## 共用搜尋空間：MTP／Draft N 已納入契約

已新增 `schemas/speculative-search.schema.json`、`tools/plan_speculative_search.py` 和可選的 Profile／Result 專用欄位。現在用 **完整 Workload Profile**（`--profile`）即可不載入模型預覽 `off`、內建／外掛 MTP 的 Draft N 候選（例如 1、2、4、8），並保留欠缺的相容性證據；沒有任何 MTP 自動執行或效能宣稱。搜尋與結果格式集中在一處，每筆候選都有 SHA-256 `candidate_id`，量測後需能由 Run Record 追溯；缺乏完整模型雜湊紀錄時，MTP 仍保持不可執行及待驗證。避免未來 UI、Runner 和 Benchmark 各自設計欄位。詳見 [變數與實作路線圖](docs/specification/tuning-roadmap.md)。

## 未來調校變數：MTP／Speculative Decoding

已將內建 NextN/MTP Head、外掛 MTP Head 和一般 Draft Model 分為不同候選，GGUF Inspector 現可唯讀擷取 `{arch}.nextn_predict_layers` 與 tokenizer metadata 摘要。`tools/verify_model_pair.py` 可重驗 target/draft 完整 SHA、單檔 GGUF、架構與 tokenizer metadata；**通過仍不代表引擎相容或模型可載入**。後續將以 `off` 基準、draft-aware 容量檢查與實際 Generation/Server Benchmark 比較。詳細規格見 [MTP／Speculative Decoding](docs/research/mtp-speculative-search.md)。

## 模型載入可行性檢查（預設不執行）

`tools/run_feasibility.py` 先驗證候選並建立只綁定 `127.0.0.1`、關閉 UI、固定 Context／Concurrency 並強制 `--fit off` 的短時 Server 載入計畫。預設只輸出計畫；只有明確加上 `--execute` 才會啟動本機 llama.exe 載入所選模型，並在 `/health` 就緒或逾時後清理程序。它不送出推論請求、不做速度測試；執行報告會每 500 毫秒取樣 ATYL 所啟動程序樹的 Host Working Set／RSS，這是取樣到的最高值，可能漏掉短暫尖峰；VRAM 峰值仍未知。操作限制與下一階段見 [MTP／Speculative Decoding](docs/research/mtp-speculative-search.md)。

## 穩定性與安全重構

所有 JSON 輸出工具統一使用原子式安全寫入：**預設不覆寫已存在的 JSON**，重跑相同輸出檔名需加 `--overwrite`，且永遠禁止把重要輸入檔當輸出。估算若子程序逾時、失敗或輸出過量，即使保存了診斷 JSON，也會以**退出碼 2**結束（0 代表預覽或估算執行完成；1 代表驗證或檔案寫入失敗）。Runtime 快照有結構 Schema 驗證，所有引擎呼叫共用受輸出／時間限制的子程序執行器；逾時、取消或 Ctrl+C 時會嘗試清理由 ATYL 啟動的程序樹（不管理外部服務）。

GGUF 快速 Fingerprint 只雜湊 Metadata 前綴加上檔案大小／修改時間；需要完整模型內容 SHA-256 時，才明確加 `--full-model-sha256`（會對大型 GGUF 進行完整磁碟讀取）。[穩定性重構與操作注意](docs/research/stability-review.md)。

## GGUF 模型資訊檢查（不啟動 llama）

```powershell
python tools/inspect_gguf.py --model "D:\AIModels\Author\Repository\model.gguf" --output model-metadata.json
```

可以只掃描 GGUF 標頭、特定架構欄位及張量描述（不讀權重），為下一階段的安全 Fitting 做資料準備。**不會在此推估模型可用 VRAM，或宣稱特定 Backend 相容。** 詳見 [GGUF Metadata Inspector v0.1](docs/research/gguf-metadata-inspection.md)。

## 官方 Fit 估算值的結構化解析

已經以真實 Windows build 11574 確認 `Vulkan0` 與 `Host` 的 Model／Context／Compute MiB 三欄格式。新估算輸出增加 `estimate.parsed_numeric_memory`（資料契約 v0.3.0）；舊版結果可使用 `tools/parse_fit_record.py` **離線轉換**，無須重新讀取 GGUF。詳細見 [Fit 記憶體解析](docs/research/fit-memory-output.md)。

## 日常 vs 低背景負載（不自動停止程式）

`fit_estimate.py --execute` 現在會保留估算前後的系統 RAM、GGML GPU 記憶體餘量與環境標籤；可分別執行 `--scenario daily`／`--scenario reduced_background`，再用 `tools/compare_scenarios.py` 比較。**這些都是當下可用量，並非模型峰值或實測吞吐量。** 詳見 [環境觀察與比較](docs/research/environment-observation.md)。

## 官方 Fit 記憶體估算：先預覽，再明確執行

已加入 `tools/fit_estimate.py`。**預設只產生估算計畫，不會啟動 llama.exe**。必須提供 `fit-params` 實機探測 JSON，若指定 GPU，還需要相同執行檔版本的裝置探測 JSON。真正讓官方引擎讀取 GGUF 及估算記憶體，必須明確加上 `--execute`。請先看 [Windows Fit 估算說明](docs/research/fit-estimate-preview.md)；**不會自動搜尋最優 GPU 層數或啟動 Server**。

## Windows 實機回歸證據

根據使用者提供的官方 llama.app build 11574 三份唯讀探測輸出，已核對 Serve／Bench／Fit 子命令與裝置列舉、整理去識別化回歸測試。差異診斷會將上游條件式 `--rpc` 與文件版本缺口分開，而非直接認定不支援。更多細節見 [Windows Build 11574 回歸與 Fit 參數基準](docs/research/windows-build11574-fit-reference.md)。

## 初步設計原則

1. 先定義工作負載與成功指標，再搜尋參數。
2. 將記憶體可行性與效能最佳化分成不同階段；不可行的組合不進入效能排名。
3. 每筆結果都記錄 llama.cpp 版本、建置後端、模型識別、硬體、完整參數、負載與重複量測。
4. 區分上游直接支援的能力、ATYL 推導出的實驗策略，以及尚未驗證的假設。
5. 保留使用者明確指定的參數；不默默改變模型品質或服務語意。
6. 以可重現、可撤回的候選設定為輸出，而非直接覆寫使用者環境。

## 先閱讀

建議先看 [參數空間規格](docs/specification/parameter-space.md)，再看 [量測方法](docs/research/benchmark-methodology.md)。上游文件持續變動；本 repo 的來源紀錄會標出檢視時的 llama.cpp commit。

## 狀態

核心資料探測與受控估算已形成 CLI 研究原型，使用者先前已在 Windows Vulkan 設備驗證單一記憶體估算；但自動調參、實際模型載入驗證及正式 Benchmark Runner 仍未完成。
