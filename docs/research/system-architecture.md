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
