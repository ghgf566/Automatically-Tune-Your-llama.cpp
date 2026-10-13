# 量測與實驗方法（初稿）

## 原則

每筆性能數據都要對應到完整且可重現的實驗條件。單純比較兩個 tokens/s 數值不足以判斷某設定較好：prompt processing、token generation、併發服務與延遲目標代表不同工作負載。

上游 `llama-bench` 會對每個測試重複執行並輸出平均 tokens/s 與標準差，支援 Markdown、CSV、JSON、JSONL、SQL 等輸出格式。其 README 也註明該工具測量不包含 tokenization 與 sampling 時間。因此它適合比較核心推論路徑，不應被當作完整 server 端到端指標。

## 工作負載描述

一個 workload profile 至少包含：

- 模型檔案識別（路徑可做本機私有值；保存檔案雜湊、架構、量化和檔案大小）。
- llama.cpp commit、binary/build 版本、backend、編譯選項與 driver/runtime 版本。
- 硬體清單與每個裝置的可用/總記憶體。
- prompt 長度與生成長度（固定值或分布）、上下文深度、請求數、併發數。
- API / 功能路徑（例如 chat、completion、embedding、多模態；啟用的工具或格式約束）。
- sampling / quality contract、服務限制及允許 ATYL 修改的參數。
- 目標指標及門檻，例如 prompt tokens/s、generation tokens/s、p50/p95 latency、穩態 throughput、峰值記憶體餘裕。

沒有使用者提供的代表性流量時，先用透明的合成 workload 作為探索條件，並標記結果不能代表實際服務。

## 分階段實驗

1. **環境盤點**：記錄 build、裝置、模型與目前參數；列出上游可用功能。
2. **容量/可行性**：排除無法載入、記憶體餘裕不足或不符合 context / concurrency contract 的設定。可把上游 fit 工具當一項候選產生器，另存其輸出。
3. **基準線**：以固定配置測試 prompt processing、token generation；server 需求則另跑代表性的端到端情境。
4. **受控探索**：先一次改動單一參數群組或採明確的有限實驗設計，避免把多個同時變更錯誤歸因給單一參數。
5. **複測與比較**：對候選和基準線使用一致的 warm-up、重複次數、prompt、併發與時長；保存原始輸出及離散程度。
6. **報告建議**：只針對該 profile 回報候選、效果、變異、限制和回復方式；保留 Pareto 選項，不把單一指標最大化視為普遍最佳。

## llama-bench 的適用範圍

README 所述三種測試是 prompt processing (`pp`)、text generation (`tg`) 以及 prompt processing + generation (`pg`)；可透過多組參數產生測試組合。測試的平均值與標準差應連同每次重複結果保存（輸出格式若有提供）。

注意：

- 將 `-p`、`-n`、`-d` 與 batch/thread/device 選項固定在比較矩陣中，說明量測的是哪段工作。
- 明確處理 warm-up、背景負載、電源/時脈變化及首次載入效應；這些控制需依平台確認。
- fit 參數可能改變有效配置，應記錄其產生的參數，而不只保存命令列輸入。
- `llama-bench` 本身不包含完整的 HTTP/API 路徑、請求排隊、tokenization 與 sampling；server 部署目標需配合端到端測試。
- 上游的 server benchmark 文件可作為後續來源，但不屬於本輪約定的五份起始來源，未在此規格中宣稱其結果或工具細節已驗證。

## 建議結果紀錄欄位

`run_id`, timestamp, source commit, binary/build id, backend, device inventory, model fingerprints, workload profile, requested parameters, effective parameters, fit output/status, warm-up/repetitions, raw result references, mean, standard deviation, latency percentiles (when available), throughput, peak host/device memory, exit/error state, notes.

機器規格、模型路徑或原始流量可能敏感；分享報告前應允許遮蔽路徑與請求內容，但不應丟掉用於重現的非敏感識別資訊。
