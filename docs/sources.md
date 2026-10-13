# 上游來源與版本紀錄

## 固定的上游快照

本輪來源檢視以 `ggml-org/llama.cpp` 的 `master` HEAD commit [`77c4540ce4667f4577753fe034a3a8042f510b8f`](https://github.com/ggml-org/llama.cpp/commit/77c4540ce4667f4577753fe034a3a8042f510b8f) 為準，檢視日期為 2026-10-13。文件內容可能在此 commit 之後變更；避免以 moving `master` 上的現況倒推此規格的檢視基準。

| # | 來源 | 用途 | 解讀界線 |
| --- | --- | --- | --- |
| 1 | [llama-server README](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/tools/server/README.md) | server 能力、CLI 參數表、環境變數與服務運行選項 | 有旗標不等於適用於所有 build/model/backend；仍需條件化 |
| 2 | [llama-bench README](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/tools/llama-bench/README.md) | bench 的參數、測試類型、重複量測與輸出格式 | 文件指出不含 tokenization / sampling 時間；不能代替端到端服務測試 |
| 3 | [Automatic parameter fitting 討論 #18049](https://github.com/ggml-org/llama.cpp/discussions/18049) | fit 的動機、記憶體調整策略、使用者顯式值的處理背景 | 討論發布於 2025-12-15；它是設計沿革，不是當前 CLI 的唯一規格 |
| 4 | [Speculative decoding 文件](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/docs/speculative.md) | 可用 speculative decoding 方法、前置條件、相關旗標與接受統計 | draft 方法及相容性會演進；不能把一種方法的結果泛化至其他 workload |
| 5 | [GGML operations / backend support](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/docs/ops.md) | operation 對不同 backend 的支援狀態與其定義方式 | 支援表不是效能數據、不是完整量化/shape 相容表，也不是調參旋鈕 |

## 檢視注意事項

- #18049 當時指出 `llama-bench` 尚未支援 fit 相關旗標，並展示了以 fit 輸出餵給 bench 的 workaround。固定快照中的最新 `llama-bench/README.md` 已列出 `--fit-target` 和 `--fit-ctx`。因此，**討論的歷史陳述不能當作目前狀態**。
- 固定快照的 server README 將 `--fit` 描述為可啟用/停用的記憶體 fit 功能；其意義不等於完整效能搜尋。
- GGML ops 支援狀態可用於理解 backend 能力，需結合實際模型圖與本機量測。
- 本輪已檢視五份約定來源；本 repo 所有「初步架構」與「搜尋方式」均為 ATYL 研究提案，並非上述上游文件已承諾的功能。

## 官方整合式安裝通道（ATYL 入口決策）

- [官方 Windows PowerShell 安裝腳本](https://llama.app/install.ps1)（非固定快照，2026-10-14 核對）：探測 CUDA → ROCm → Vulkan → CPU 並將 `llama.exe` 安裝至 LocalAppData 的 WindowsApps；不能預先假定選中何種 GPU 後端。
- [固定上游 commit 的整合式指令路由原始碼](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/app/llama.cpp)：確認 `version`、`serve`、`bench`、`fit-params` 與 `update` 指令。`update` 僅官方 installer build 支援。
- 本項是 ATYL 的執行檔發現和路由規則，**不改寫**先前五份研究文件所使用的固定 CLI 參數基準。
