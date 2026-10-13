# ATYL — Automatically Tune Your llama.cpp

ATYL 研究如何根據使用者的硬體、模型、服務負載與目標，自動找出合適的 llama.cpp 執行參數。

目前階段是**理解與規格化**：整理上游已提供的控制面、量測能力、限制與研究問題。這個 repo 尚未包含 GUI、完整 tuner、安裝器或執行期自動修改器。

## 目前文件

- [文件索引](docs/README.md)
- [能力與參數空間規格](docs/specification/parameter-space.md)
- [CLI Registry v1](docs/specification/cli-registry.md)
- [實驗 Profile／Result 契約 v1](docs/specification/experiment-contracts.md)
- [量測與實驗方法](docs/research/benchmark-methodology.md)
- [初步系統架構](docs/research/system-architecture.md)
- [上游來源與版本](docs/sources.md)

## 初步設計原則

1. 先定義工作負載與成功指標，再搜尋參數。
2. 將記憶體可行性與效能最佳化分成不同階段；不可行的組合不進入效能排名。
3. 每筆結果都記錄 llama.cpp 版本、建置後端、模型識別、硬體、完整參數、負載與重複量測。
4. 區分上游直接支援的能力、ATYL 推導出的實驗策略，以及尚未驗證的假設。
5. 保留使用者明確指定的參數；不默默改變模型品質或服務語意。
6. 以可重現、可撤回的候選設定為輸出，而非直接覆寫使用者環境。

## 先閱讀

建議先看 [參數空間規格](docs/specification/parameter-space.md)，再看 [量測方法](docs/research/benchmark-methodology.md)。上游文件持續變動；本 repo 的來源紀錄會標出檢視時的 llama.cpp commit。

## 狀態

研究起始文件。尚未實作或驗證自動調參流程，也沒有宣稱任何參數在所有硬體或模型上最佳。
