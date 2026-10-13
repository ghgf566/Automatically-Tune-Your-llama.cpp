# ATYL 文件索引

## 規格

- [CLI 能力登錄表 v1](specification/cli-registry.md)：固定來源的 302 筆 CLI 文件清單及 32 筆調校知識。
- [官方安裝與引擎入口](research/official-installation.md)：官方 Windows PowerShell 安裝通道、整合式子命令和不自動更新原則。
- [Runtime CLI 探測 v0.6](research/runtime-capability.md)：唯讀查詢本地執行檔並與固定登錄表比對。
- [Profile／Result 契約 v1](specification/experiment-contracts.md)：使用者鎖定參數、工作負載、Fit 與實際啟動狀態。
- [能力與參數空間](specification/parameter-space.md)：可控的效能變數、限制與分類。
- [上游來源與版本](sources.md)：五份起始來源及固定版本。

## 研究

- [官方 Fit 記憶體輸出解析](research/fit-memory-output.md)：實機 MiB 格式、三欄定義及舊結果離線轉換。
- [日常環境與低背景負載比較](research/environment-observation.md)：同設定量測前後系統 RAM 與 GGML GPU 可用量，不干預背景程式。
- [Fit 記憶體估算：預覽優先](research/fit-estimate-preview.md)：基於新鮮 GGUF Metadata、官方 Runtime Snapshot 的單候選估算；只有 `--execute` 會呼叫引擎。
- [GGUF Metadata Inspector v0.1](research/gguf-metadata-inspection.md)：唯讀解析模型架構、量化格式代號、張量描述與分檔提示。
- [Windows Build 11574 回歸資料與 Fit 參數研究](research/windows-build11574-fit-reference.md)：三份實機查詢證據、差異分類與 13 項精選 Fit 旗標。
- [Windows 真實安裝版 Smoke Test](research/windows-smoke-test.md)：無模型、唯讀確認 Backend／裝置資訊與 CLI 路由。
- [量測與實驗方法](research/benchmark-methodology.md)：工作負載、重複量測與公平比較。
- [初步系統架構](research/system-architecture.md)：模組責任與初步里程碑。

## 目前限制

登錄表來自官方文件，不是執行檔能力驗證。尚未開始 Benchmark Runner，也沒有自動修改使用者 llama.cpp 設定的功能。
