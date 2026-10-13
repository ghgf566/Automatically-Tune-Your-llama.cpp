# ATYL — Automatically Tune Your llama.cpp

ATYL 研究如何根據使用者的硬體、模型、服務負載與目標，自動找出合適的 llama.cpp 執行參數。

目前階段是**理解與規格化**：整理上游已提供的控制面、量測能力、限制與研究問題。這個 repo 尚未包含 GUI、完整 tuner、安裝器或執行期自動修改器。

## 官方 llama.cpp 安裝與執行入口

ATYL **優先支援官方 [llama.app](https://llama.app) 安裝的整合式 `llama.exe`**，不下載或打包 GitHub Releases 中的獨立推理引擎，也不自行替換 llama.cpp 核心。預設由 PATH 尋找 `llama`，以 `llama serve`／`llama bench` 分別對應原有的 server／bench 參數研究；既有獨立執行檔保留為明確指定時的相容模式。

在 **PowerShell** 安裝（`irm`／`iex` 不是 CMD 原生命令）：

```powershell
irm https://llama.app/install.ps1 | iex
llama version
llama help all
llama serve --help
llama bench --help
```

Windows 安裝腳本目前將實際執行檔放在 `%LOCALAPPDATA%\Microsoft\WindowsApps\llama.exe`；依可用 CUDA、ROCm、Vulkan 與 CPU 版本選擇預編譯檔。**實際選到的後端必須以本機檢查為準。** 安裝和更新由官方通道／使用者自行執行，ATYL 目前不自動執行遠端腳本或 `llama update`。

```powershell
python tools/probe_runtime.py --command serve --output server-capability.json
python tools/probe_runtime.py --command bench --output bench-capability.json
python tools/probe_runtime.py --command fit-params --output fit-capability.json
```

探測器透過統一的 Command Adapter 只呼叫版本與說明指令；上述命令不會載入模型或啟動伺服器。完整細節見 [官方安裝策略](docs/research/official-installation.md) 與 [Runtime CLI 探測文件](docs/research/runtime-capability.md)。

## 目前文件

- [文件索引](docs/README.md)
- [能力與參數空間規格](docs/specification/parameter-space.md)
- [CLI Registry v1](docs/specification/cli-registry.md)
- [Runtime CLI 探測 v0.6](docs/research/runtime-capability.md)
- [實驗 Profile／Result 契約 v1](docs/specification/experiment-contracts.md)
- [量測與實驗方法](docs/research/benchmark-methodology.md)
- [初步系統架構](docs/research/system-architecture.md)
- [上游來源與版本](docs/sources.md)

## 目前的架構分層

- `tools/engine_adapter.py`：官方整合式 CLI 與 Legacy 命令路由的唯一來源。
- `tools/engine_identity.py`：實際二進位 SHA-256、版本回報與獨立的 Registry 來源 Commit；更新版本時生成不同的能力快取 Key。
- `schemas/*.json`：已升級為 `1.1.0`，實際引擎 Build 與文件 Registry SHA 不再互相冒充。Build／Backend 尚未證實時以 `null` 保存。
- `fit-params` 目前僅支援不載入模型的 `--help` 探測，已加入 13 項精選 Fit 旗標作為能力研究基準。
- `tools/device_inventory.py`：可選擇性列舉 GGML 回報的非 CPU 裝置及可用記憶體；尚未驗證實際 Backend／模型能力。
- [Windows 手動 Smoke Test](docs/research/windows-smoke-test.md)：操作官方已安裝的 llama.exe，不需要啟動推理服務。

## Windows 實機回歸證據

根據使用者提供的官方 llama.app build 11574 三份唯讀探測輸出，已核對 Serve／Bench／Fit 子命令與裝置列舉、整理去識別化回歸測試。差異診斷會將上游條件式 `--rpc` 與文件版本缺口分開，而非直接認定不支援。更多細節見 [Windows Build 11574 回歸與 Fit 參數基準](docs/research/windows-build11574-fit-reference.md)。

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
