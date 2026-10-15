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

## Background-load controls

The opt-in fit-estimate stage captures declared `daily` and `reduced_background` environments plus pre/post headroom. These snapshots are **not benchmark measurements**, do not record peak memory, and are not sufficient to attribute changes to Wallpaper Engine or any other process. The current speculative runner controls warm-up and repetitions for fixed local prompts; it does not isolate OS page cache, thermal/power state, or background processes. Treat those as limitations when interpreting the numbers.

## MTP／Speculative Decoding 的量測邊界

完整 generation 基準以 `spec_mode=off` 對照 `embedded_mtp` 與經驗證的 `external_mtp`，保持 target model、Context、取樣契約、併發與測試文本一致；公平容量比較須保留 Draft 額外 RAM/VRAM 消耗。現有 runner 記錄 drafted/accepted token 數、接受率、decode throughput、TTFT、latency 與資源觀測；新版 sweep 為每個 workload/N 配對獨立 off baseline。外掛 MTP、長上下文、併發、使用者流量與品質等價仍未驗證。`llama-bench` 核心測試不一定覆蓋完整伺服器 speculative 路徑，因此須以 Server/CLI 實際 workload 驗證。參考 [MTP 條件式候選研究](mtp-speculative-search.md)。

## Draft N 預測數量的交互效應

對 MTP，不只記錄 `--spec-type`，也要固定／搜尋 `--spec-draft-n-max`：先在相同主模型、Context、Offload、Draft 放置及取樣條件下測不同 N（例如 1/2/4/8），再依有限預算細部搜尋；另外記錄引擎實際生效的 N（未知保持 `null`）。Acceptance Rate、每輪接受 Token 數、完整端到端 Tokens/s、TTFT、p95、Host/GPU 記憶體額外成本必須一併觀察，不能因 N 大就認定更快，也不能把 GGUF NextN 層數等同最大草稿長度。參見 [調校架構路線圖](../specification/tuning-roadmap.md)。

## CPU MoE 權重與 GPU 專家快取實測

2026-10-15 在官方整合版 CUDA build 11624（`af0579c78`）、GTX 1070 Ti 8 GiB 與 Qwen3.6-35B-A3B IQ4_XS 上完成兩輪探索。第二輪固定一般 GPU layer 16/41、CPU MoE 權重、Q4_0 K/V、context 4096、單請求、Fit 關閉、prompt RAM cache 關閉；對 0/1024/2048 MiB GPU expert cache，各跑一個 warm-up 和五次 256-token completion，seed 決定的實際順序為 0 → 2048 → 1024 MiB。平均 decode 速度分別為 **6.08、6.76、6.70 tokens/s**；相較基線增加約 **11.2%** 與 **10.3%**。中位數為 6.05、6.83、6.73 tokens/s；樣本標準差約 0.22、0.15、0.46 tokens/s（變異係數 3.6%、2.2%、6.9%）。平均 latency 約 44.74、40.26、40.52 秒。第二輪增加了生成長度與重複數，並採可重現的亂序；上一輪 3 次 128-token 結果約為 6.36、6.65、6.74 tokens/s，可作探索背景，不與第二輪合併計算。

較長矩陣仍只有單一合成 prompt、單一亂序輪次，不能排除溫度、背景負載或順序交互影響；1024 與 2048 MiB 的速度分布重疊，2048 MiB 變異也較大，因此不據此選出最佳設定。需要多種 prompt、多輪反轉或平衡順序及更多重複。實際觀察到 K/V buffer 分布在 CPU 13.50 MiB 與 CUDA 9.00 MiB；16/41 層表示一般層並非全數在 GPU。log 顯示約 967 MiB 模型 buffer 在 CUDA，MoE cache 實際配置約 1023/2046 MiB，2 GiB 案例的 CPU expert 權重約 5562 MiB。shared-expert tensor 的位置沒有獨立證據。WDDM 下程序歸因 VRAM 仍未知，系統 VRAM 讀值包含其他程序。

2026-10-15 後續 runner 的 GPU Fit target 與 free-VRAM reserve 預設降為 **512 MiB**，以提高可用顯存利用率；每次實驗仍須記錄實際配置並持續監看系統級 free VRAM。歷史實驗保留當時 1024 MiB reserve，不回寫或混算。

Qwen3.6 GGUF 未提供正值 `nextn_predict_layers`，因此不符合本專案內嵌 MTP Draft N sweep 的候選條件。MoE 快取探索和 MTP Draft N sweep 是不同實驗軸；不得把這組 MoE 速度結果解讀為 MTP 測量。
