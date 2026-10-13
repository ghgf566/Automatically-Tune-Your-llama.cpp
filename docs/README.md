# ATYL 文件索引

## 規格

- [能力與參數空間](specification/parameter-space.md)：llama-server / llama-bench 可控制的效能相關變數、限制與分類。
- [上游來源與版本](sources.md)：五份起始來源的用途、固定版本與解讀界線。

## 研究

- [量測與實驗方法](research/benchmark-methodology.md)：工作負載定義、結果欄位、重複量測與公平比較。
- [初步系統架構](research/system-architecture.md)：把設定、可行性檢查、實驗、分析與建議分開的研究架構。

## 維護規則

上游 CLI 會變動。新增參數時需記錄來源路徑、檢視的上游 commit、是否可被 llama-bench 直接量測，以及已知互斥或前置條件。若只在特定模型或 backend 有效，必須明示條件，不能把單一機器的觀察寫成通則。
