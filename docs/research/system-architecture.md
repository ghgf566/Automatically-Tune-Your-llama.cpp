# 初步研究架構

這是為未來調參研究提出的責任邊界，不是已實作系統，也不是已決定的程式語言或 GUI 架構。

## 資料流

```text
環境與使用者限制
        │
        ▼
Profile（硬體、build、模型、工作負載、目標）
        │
        ▼
Engine Discovery / Command Adapter（官方 PATH llama → serve／bench／fit-params）
        │
        ▼
Capability Registry（上游旗標、支援條件、來源 commit）
        │
        ▼
候選產生 ────── 使用者鎖定值 / 搜尋界線
        │
        ▼
可行性閘門（記憶體、相容性、context、功能需求）
        │
        ▼
Runner（隔離執行、逾時、取消、資源觀察）
        │
        ▼
Benchmark Adapter（llama-bench 與未來 server workload）
        │
        ▼
結果儲存（設定、環境、原始結果、錯誤、統計）
        │
        ▼
分析與建議（依 profile 呈現多目標取捨，可重現、可回復）
```

## 元件責任

- **Profile**：描述比較所需的輸入條件與固定限制；應能匯出為版本化、可審閱的文字資料。
- **Engine Discovery / Command Adapter**：優先從 PATH 使用官方 `llama.exe`，映射 `serve`／`bench`／`fit-params` 子命令，區分 binary SHA-256、實際引擎版本與 Registry 文件 Commit；舊獨立執行檔僅作明確指定的相容模式。不自行安裝或修改引擎。
- **Capability Registry**：描述每個旗標的名稱、型別、範圍、預設、適用程式、前置條件、互斥、風險與上游證據。資料需綁定 llama.cpp commit，避免假設 CLI 長期不變。
- **Candidate Generator**：從有限且受限的搜尋空間產生候選。初期不預設採用任何一種搜尋演算法。
- **Feasibility Gate**：先檢查模型/裝置/功能相容性與記憶體條件；執行期錯誤也必須被保存，而不是當成慢速結果。
- **Runner**：啟動可控的 llama.cpp 命令、捕捉標準輸出/錯誤、設定 timeout，並於結束後確認子程序已停止。預設應避免修改既有安裝或服務。
- **Benchmark Adapter**：將 profile 映射至明確測試情境，解析 llama-bench 原始輸出；服務目標需要獨立的 HTTP 工作負載 adapter。
- **Result Store**：保存 requested/effective config、版本、環境、原始結果與錯誤，支援去重和重現。
- **Analyzer**：依使用者目標比較可行候選，顯示變異與 trade-off；產生建議設定與啟動命令，留由使用者檢視/採用。

## 已確定的引擎來源策略

- 使用官方 llama.app 的整合式 CLI（`llama.exe`）作為 Windows 第一優先入口；ATYL 不管理上游 Release ZIP，不綁定安裝位置。
- Registry 保持邏輯名稱 `llama-server`／`llama-bench`，但由指令分派器轉為 `llama serve`／`llama bench`；Fit Help 由 `llama fit-params --help` 觀察，暫無固定 Fit 旗標 Registry。
- 安裝器、GPU Backend 探測及自我更新交由官方和使用者；ATYL 只讀取並驗證其實際能力。工具變更後依版本／雜湊清除快取。

## GGUF 模型可行性資料階段

新增 `tools/inspect_gguf.py` 作為獨立模型中繼資料入口：只讀取 GGUF 標頭、KV 與張量描述，保留檔案局部性與分檔狀態；不讀權重、不啟動推論引擎。解析出來的模型資訊不能直接與 GGML 裝置剩餘記憶體比較，因為模型載入、KV Cache、Context 和 Backend Buffer 的需求仍未知。已加入單一候選參數的 `tools/fit_estimate.py`：預設預覽，只有明確 `--execute` 才會讓官方引擎讀取模型並列印估算。它驗證 Fit Help 與設備快照的 SHA-256 相符，仍不代表實際載入成功或找到最佳配置。已針對 Windows 實機輸出實作 Model／Context／Compute 的嚴格 MiB 解析，並支援舊 JSON 離線轉換；尚待多候選搜尋、完整分檔確認、實際載入與效能驗證。

## 測試情境與可用記憶體記錄

現在估算 Runner 可以在 `--execute` 明確授權後，各在開始前和結束後擷取一次系統 RAM 與 GGML 記憶體餘量；用 `daily`／`reduced_background` 保存手動實驗條件。這不是高頻記憶體追蹤或載入成功率，也不能把前後差值歸因於模型或 Wallpaper Engine。跨情境比較器只接受同引擎版本、同記錄模型與同候選設定。

## 穩定性共用模組

- `tools/safe_output.py`：原子 UTF-8 輸出，預設禁止覆寫，另外保護模型、能力快照、執行檔與資料來源路徑。
- `tools/runtime_snapshot.py`：對產生及消費的 Runtime JSON 做結構驗證，SHA 不一致就拒絕。
- `tools/process_runner.py`：共用逾時、輸出擷取與磁碟輸出限制，絕不經 shell；目前殺的是直接子程序，目前透過 `process_lifecycle.py` 對 ATYL 建立的 Session/Process Group 進行 best-effort 程序樹清理，取消與逾時皆會清理直接子程序及可追蹤的後代；完整 OS Job Objects／長期 Service 容器管理仍待後續。
- `tools/model_identity.py`：預設低 I/O 模型 Metadata Prefix Hash；使用者可指定完整 SHA-256，但它也不等於權重內容正確可推論的證明。
- GitHub CI 同時在 Windows／Linux、Python 3.11／3.13 跑模擬與平台檢查；仍無真 GPU、模型載入測試。

## 裝置探測的狀態界線

目前可選擇性以 `--list-devices` 查詢 GGML 列出的**非 CPU** 裝置及其記憶體，並保留成功、沒有列出與未知三種狀態。這只屬於引擎環境觀察，尚未驗證 CPU／NPU 的可運算能力、後端算子支援或特定模型能否載入。真正的可行性閘門不能只靠該輸出。

## Speculative Decoding 的結構化調校路線

Profile 可透過 `speculative_search` 引用共用 Speculative Schema，描述 MTP 開關／模式、`draft_n_max_candidates`、外掛 Draft 識別與獨立 Draft Placement。`plan_speculative_search.py` 目前僅產生有限候選與未驗證證據狀態；尚未執行任何 MTP。將來同一份 Result Schema 的 `speculative_observation` 記錄 Requested vs Effective N、接受率、吞吐、TTFT、尾延遲與記憶體，避免將數據直接混入未經驗證的參數字典。詳見 [調校變數的相依圖與階段規劃](../specification/tuning-roadmap.md)。

## 尚未決策

- Profile／Result 已有研究階段的 JSON Schema `1.1.0`，未來的正式資料遷移規則與 API 相容承諾仍待決策。
- CPU-only、多 GPU、NUMA、RPC、UMA 及不同 backend 的支援順序。
- 記憶體容量採上游 `--fit`、ATYL 量測或兩者組合的界線。
- 是否需要 server 端到端 adapter、用何種 workload 和外部工具。
- 搜尋演算法、暫停/恢復策略與跨機器結果可比性。
- 安全的程序隔離、使用者權限與模型/資料隱私。
- GUI 是否有需求；目前只先保留元件邊界，不開始設計 GUI。

## 最小研究里程碑

1. 完成固定上游版本的 CLI capability registry。
2. 定義 workload/profile 與結果 schema。
3. 以人工指定候選驗證 llama-bench 輸入、輸出與重複量測。
4. 驗證容量失敗、timeout、程序清理和 effective config 紀錄。
5. 再決定自動搜尋策略，並在明確需求後討論 UI。
