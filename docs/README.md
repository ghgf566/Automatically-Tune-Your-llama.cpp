# ATYL 文件索引

## 規格

- [調校變數架構與實作路線圖](specification/tuning-roadmap.md)：正式納入 MTP Draft N、外掛模型與其他配置的相依關係、驗證階段及搜尋策略。
- [CLI 能力登錄表 v1](specification/cli-registry.md)：固定來源的 302 筆 CLI 文件清單及 32 筆調校知識。
- [官方安裝與引擎入口](research/official-installation.md)：官方 Windows PowerShell 安裝通道、整合式子命令和不自動更新原則。
- [Runtime CLI 探測 v0.6](research/runtime-capability.md)：唯讀查詢本地執行檔並與固定登錄表比對。
- [Profile／Result 契約 v1](specification/experiment-contracts.md)：使用者鎖定參數、工作負載、Fit 與實際啟動狀態。
- [能力與參數空間](specification/parameter-space.md)：可控的效能變數、限制與分類。
- [上游來源與版本](sources.md)：五份起始來源及固定版本。

## 研究

- [Feasibility 前最終驗收](research/pre-feasibility-audit.md)：跨模組檢查結果、已修正的證據缺口與進入實際載入階段的明確界線。
- [MTP／Draft 配對與載入可行性](research/mtp-speculative-search.md)：重驗 target/draft 身分；預設只建立計畫，另以明確 `--execute` 進行受控本機載入檢查。
- [MTP／Speculative Decoding 研究變數](research/mtp-speculative-search.md)：區分內建 MTP、外掛 MTP、一般 Draft Model；定義可行性驗證與公平 Benchmark。
- [安全性與穩定性整理](research/stability-review.md)：安全輸出、退出碼、快照、共用 Process Runner、Fingerprint、Windows CI。
- [官方 Fit 記憶體輸出解析](research/fit-memory-output.md)：實機 MiB 格式、三欄定義及舊結果離線轉換。
- [日常環境與低背景負載比較](research/environment-observation.md)：同設定量測前後系統 RAM 與 GGML GPU 可用量，不干預背景程式。
- [Fit 記憶體估算：預覽優先](research/fit-estimate-preview.md)：基於新鮮 GGUF Metadata、官方 Runtime Snapshot 的單候選估算；只有 `--execute` 會呼叫引擎。
- [GGUF Metadata Inspector v0.1](research/gguf-metadata-inspection.md)：唯讀解析模型架構、量化格式代號、張量描述與分檔提示。
- [Windows Build 11574 回歸資料與 Fit 參數研究](research/windows-build11574-fit-reference.md)：三份實機查詢證據、差異分類與 13 項精選 Fit 旗標。
- [Windows 真實安裝版 Smoke Test](research/windows-smoke-test.md)：無模型、唯讀確認 Backend／裝置資訊與 CLI 路由。
- [量測與實驗方法](research/benchmark-methodology.md)：工作負載、重複量測與公平比較。
- [MoE 專家快取掃描器](../tools/run_moe_cache_sweep.py)：CPU MoE 權重、GPU 專家快取、KV 型別與實機測量契約。
- [初步系統架構](research/system-architecture.md)：模組責任與初步里程碑。

## 目前限制

登錄表來自官方文件，不是執行檔能力驗證。受控載入檢查本身不執行推論；另有明確 `--execute` 閘門的 off/MTP Server benchmark、bounded Draft N sweep 和 MoE GPU expert-cache sweep。Qwen3.8-27B 已完成 Draft N 1/2/4 × 三種合成 workload 的完整 9 組真實硬體 sweep，結果與限制見 [MTP sweep 報告](research/mtp-speculative-search.md)。Qwen3.6-35B-A3B 不含內嵌 NextN metadata，不能用於該 MTP sweep。Qwen3.6 的首輪 MoE cache sweep 已完成，但只有單一合成 prompt、每設定 3 次 128-token 重複，仍需擴充工作負載與重複數。後續 GPU Fit target/free-VRAM reserve 預設為 512 MiB。Runner 會記錄受控程序樹 Host Working Set／RSS 取樣值，並在 NVIDIA GPU 上觀察系統層級記憶體；WDDM 程序 VRAM 可能未知，系統讀數也不會映射到 llama 裝置 ID。
