# llama.cpp 能力與參數空間（初稿）

**基準來源：** llama.cpp `ggml-org/llama.cpp` commit [`77c4540ce4667f4577753fe034a3a8042f510b8f`](https://github.com/ggml-org/llama.cpp/commit/77c4540ce4667f4577753fe034a3a8042f510b8f)，檢視日期 2026-10-13。來源與限制見[來源登錄表](../sources.md)。

此文件建立研究用的分類與語意，不是所有旗標的完整逐字清單。CLI 名稱、預設值及相容性以指定 commit 的上游文件為準。部分參數由 build/backend/model 決定，不能視作通用連續變數。

## 1. 調參問題的邊界

ATYL 的目標是針對明確指定的模型、建置、硬體與工作負載，探索安全且可重現的 llama.cpp 執行設定。至少要把以下因素分開：

- **環境條件**：CPU、RAM、NUMA、GPU / accelerator、各裝置可用記憶體、驅動、llama.cpp commit、編譯選項與 backend。
- **模型條件**：GGUF 檔案、量化、模型架構、tensor 數量與大小；draft 模型或 projector 等輔助檔案另行識別。
- **工作負載**：prompt 長度分佈、生成長度、併發、上下文深度、請求/API 類型與是否需要工具或多模態輸入。
- **目標與限制**：吞吐、首 token 延遲、單請求延遲、記憶體餘裕、品質 / 語意限制，以及允許改動的參數。
- **設定值**：保留使用者固定值，將未指定值視為候選；必須辨別上游自動推導值和使用者明確值。

沒有這些條件時，「最快參數」沒有可比較的意義。

## 2. 主要效能控制面

下表列出研究優先級高的控制群組。名稱為上游常見旗標示例，不代表每個 benchmark binary 都支援完全相同的介面。

| 群組 | 常見旗標 | 調整意義 / 主要影響 | 初步處理 |
| --- | --- | --- | --- |
| CPU 執行緒與親和性 | `--threads`, `--threads-batch`, `--cpu-mask`, `--cpu-range`, `--cpu-strict`, `--poll`, `--numa` | CPU decode、prompt processing、核心放置與 NUMA 行為 | 依 CPU 與負載條件探索；batch 與 generation 執行緒分開記錄 |
| 模型裝置放置 | `--device`, `--n-gpu-layers` / `--gpu-layers`, `--split-mode`, `--tensor-split`, `--main-gpu` | CPU/GPU 混合執行、跨 GPU 分層或分割；改變記憶體位置與傳輸成本 | 先做裝置/記憶體可行性檢查；多 GPU 參數需依裝置順序保存 |
| Tensor / MoE 放置 | `--override-tensor`, `--cpu-moe`, `--n-cpu-moe`, `--n-cpu-ffn`, `--moe-cache-mib` | 個別 tensor、MoE expert 或 FFN 的放置與快取 | 高度依模型架構；預設不進入第一階段廣泛搜尋 |
| Context / KV | `--ctx-size`, `--cache-type-k`, `--cache-type-v`, `--kv-offload`, `--kv-unified`, `--kv-unified-per-slot`, `--cache-ram` | 可用上下文、KV 記憶體占用與裝置放置 | Context 同時是容量限制與工作負載規格；不得為了跑分而偷偷縮短需求 |
| Prompt 計算批次 | `--batch-size`, `--ubatch-size`, server `--parallel` / slot 與 batching 選項 | Prompt processing 計算效率、暫存記憶體及併發 | 依 prompt 長度與併發共同評估；不能只看單請求 |
| Attention / backend offload | `--flash-attn`, `--op-offload`, `--no-host`, `--repack`, `--load-mode`, `--lazy-mode` | 運算 kernel、buffer 路徑、權重載入/常駐行為 | 依 build/backend 和模型支援狀態進行條件式測試 |
| Speculative decoding／MTP | `--spec-type`（`none`／`draft-mtp`／`draft-simple`）、`--spec-draft-n-max`、選用外掛 `--model-draft`、草稿裝置／層數 | 內建或外掛的 MTP 與一般 Draft Model 是不同候選；接受率、VRAM/RAM 額外成本會影響整體收益 | 先確認 GGUF 的 NextN/MTP 欄位與引擎支援，再以同負載比較開／關；[詳見 MTP 搜尋規格](../research/mtp-speculative-search.md) |
| 模型/精度選擇 | GGUF 模型與量化檔、draft/projector 檔案 | 改變權重大小、記憶體需求、精度與 kernel 路徑 | 初期視為實驗條件而非同一模型下的 runtime 微調參數 |
| 取樣與輸出語意 | sampler、temperature、top-k/top-p 等 | 主要定義生成分佈與輸出語意，也可能影響服務工作量 | 預設鎖定為 workload contract，不以效能名義改動 |

上游 `llama-server` README 是 server CLI 的主要能力索引；`llama-bench` README 則列出 benchmark 可枚舉的測試參數。對調參器而言，server 選項「存在」不代表 benchmark 可以完整評估其真實服務效應。

## MTP 也是條件式搜尋變數

MTP 至少有「關閉」、「主模型內建 NextN/MTP Head」、「外掛相容 MTP Head」三種模式；另外的一般小 Draft Model（`draft-simple`）不得混為 MTP。是否啟用、草稿 Token 數與外掛模型的放置都可能改變最佳 Offload 與記憶體可行性。嚴禁僅依檔名宣稱支援，須以 `{arch}.nextn_predict_layers`、額外 GGUF 配對資料、當前引擎 Help 及後續實際載入驗證逐層確認。詳見 [MTP 與 Speculative Decoding 條件式調校規格](../research/mtp-speculative-search.md)。

## 3. 上游自動 fit 與 ATYL 的差異

上游已有 `--fit` / `--fit-target` / `--fit-ctx` 與 `llama-fit-params` 能力，會根據裝置可用記憶體調整部分記憶體相關設定。#18049 討論記錄了其設計動機和早期策略：先試原設定，再降低 context，仍不足時將 tensor 從 VRAM 移至 RAM，並優先保留 dense tensors。

這種 **memory fit** 和 ATYL 想研究的 **performance tuning** 是兩個不同問題：

1. fit 回答「怎麼讓指定模型/configuration 在記憶體裡可載入」；
2. tuner 還需回答「在固定品質、上下文與負載要求下，哪些可行配置的量測表現較好」。

現行 `llama-bench` README 已列出 `--fit-target` 和 `--fit-ctx`，所以 #18049 內「bench 尚未支援 fit」是歷史描述，不應當作目前狀態。fit 結果、原始設定與實際啟動結果要分開保存。

## 4. GGML backend 能力的解讀

上游 `docs/ops.md` 是 GGML operation 對各 backend 的支援表，狀態包括完全支援、部分支援與不支援。這張表可協助解釋某個模型圖在特定 backend 上可能走到的運算路徑或限制，但它：

- 不是模型層級的效能保證；
- 不是完整的 dtype / quantization / shape 組合矩陣；
- 不是 llama-server 可調參數列表；
- 不能單獨預測某配置的 tokens/s 或延遲。

ATYL 應把 backend 與 build capability 作為環境描述和候選過濾依據，再以本機測量驗證，不應直接從 ops 支援符號推算排名。

## 5. 變數分類與搜尋約束

每個設定在資料模型中標為以下一類：

- **固定條件**：由使用者鎖定；不可自動改寫。
- **條件式候選**：只有設備、backend、模型或工作負載滿足前置條件時才探索。
- **離散候選**：例如 split mode、Flash Attention on/off/auto、cache type。
- **數值候選**：例如 threads、batch size、context size、GPU layer count；使用有限且有界的候選集合。
- **外部選擇**：模型/量化/draft model 等，會改變比較對象，需分組報告。
- **暫不調整**：會改變輸出品質、功能或安全語意的設定，例如 sampling 與 API 行為。

搜尋每一步需保留原始設定，記錄覆蓋原因與衍生出的有效值。先排除無效/超出需求的配置，再做效能比較。不得將單一硬體上的最佳值提升為全域預設。

## 6. 初版能力清單待補事項

後續以固定 llama.cpp commit 逐項盤點：

- server 與 bench 的完整旗標、型別、預設、別名、互斥/依賴關係；
- 對應環境變數和設定檔可覆寫行為；
- 各參數在不同 backend / model family 的有效性；
- server benchmark 中的請求併發、prompt 分佈與 latency 指標；
- 自動 fit 對顯式使用者值的保留規則、結果如何傳回呼叫端；
- 每個變數是否可在安全的隔離環境中自動修改與回復。
