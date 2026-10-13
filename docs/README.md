# ATYL 文件索引

## 規格

- [CLI 能力登錄表 v1](specification/cli-registry.md)：固定來源的 302 筆 CLI 文件清單及 32 筆調校知識。
- [官方安裝與引擎入口](research/official-installation.md)：官方 Windows PowerShell 安裝通道、整合式子命令和不自動更新原則。
- [Runtime CLI 探測 v0.4](research/runtime-capability.md)：唯讀查詢本地執行檔並與固定登錄表比對。
- [Profile／Result 契約 v1](specification/experiment-contracts.md)：使用者鎖定參數、工作負載、Fit 與實際啟動狀態。
- [能力與參數空間](specification/parameter-space.md)：可控的效能變數、限制與分類。
- [上游來源與版本](sources.md)：五份起始來源及固定版本。

## 研究

- [量測與實驗方法](research/benchmark-methodology.md)：工作負載、重複量測與公平比較。
- [初步系統架構](research/system-architecture.md)：模組責任與初步里程碑。

## 目前限制

登錄表來自官方文件，不是執行檔能力驗證。尚未開始 Benchmark Runner，也沒有自動修改使用者 llama.cpp 設定的功能。
