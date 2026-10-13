# Windows 官方 llama.app Smoke Test（手動、唯讀）

ATYL 目前的 GitHub CI 僅使用 mock subprocess 和合成硬體輸出。以下測試需在**使用者自行安裝、可信任的官方 llama.exe** 上操作，不會下載模型、啟動伺服器或執行 Benchmark。

## 事前檢查

在 Windows **PowerShell** 開啟 ATYL 的專案根目錄。需要 Python 3.9+；如未安裝 jsonschema：

```powershell
python -m pip install -r requirements-dev.txt
Get-Command llama -All
llama version
llama help all
llama serve --help
llama bench --help
```

如果 `Get-Command llama` 指到不熟悉的位置，請先確認來源，**不要交給探測器執行**。Windows 官方安裝通常會在 `$env:LOCALAPPDATA\Microsoft\WindowsApps\llama.exe`。ATYL 預設透過 PATH 發現，並不假定安裝路徑。

## 依序進行 ATYL 唯讀探測

```powershell
python tools/probe_runtime.py --command serve --output server-capability.json
python tools/probe_runtime.py --command bench --output bench-capability.json
python tools/probe_runtime.py --command fit-params --output fit-capability.json
```

這三個操作只查詢版本、全命令列表和各子命令的 Help。執行完確認 JSON 中 `identity.binary_sha256`、`identity.engine_version_output`、`entrypoint.command`，以及 `command_results.help` 的狀態；`identity.engine_build_commit` 尚未做可靠解析，可能是 `null`。

## 需要測試 GPU／加速器列舉時再選擇執行

```powershell
python tools/probe_runtime.py --command serve --include-devices --output devices-capability.json
```

這個額外步驟會呼叫 `llama serve --list-devices`，可能初始化 GPU 驅動。ATYL 只有在本機 `serve --help` 確實列出該旗標時才會呼叫。`fit-params` 不提供 `--include-devices` 選項。

官方固定來源 [common/arg.cpp 的 `common_print_available_devices`](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/common/arg.cpp#L1148-L1166) 列印：

```text
Available devices:
  CUDA0: Example GPU (8192 MiB, 4096 MiB free)
```

上面是格式示例，**不是你電腦的測量結果**。ATYL 會把 `device_inventory.status` 記為 `observed`、`none` 或 `unknown`，其中 `none` 代表沒有列舉到非 CPU 裝置，`unknown` 代表無法可靠判斷。資料欄位的約束見 [Device Inventory JSON Schema](../../schemas/device-inventory.schema.json)。

## 檢查重點與限制

- 同一份輸出必須能區分：實際引擎版本／Binary SHA-256、固定 CLI Registry 版本。
- 若 `--list-devices` 沒在 Help 裡，不能直接認定沒有 GPU。查詢失敗、逾時、格式變更都記錄為 `unknown`。
- `total_memory_mib`／`free_memory_mib` 是 GGML 回報的**查詢當下**數字，不是 ATYL 對 VRAM 的獨立實測，也不保證一定可用於模型。
- 該命令刻意**不列出 CPU**；也無法單憑輸出判定 NPU 能否用、模型算子是否支援、記憶體是否足夠或是否真正有推理效能。
- 若要分享 JSON，請先檢查版本摘要、裝置描述與本機訊息是否包含隱私資訊。
- 不需要執行 `llama update`；ATYL 不會安裝、修改或重啟 llama.cpp。

已有一組官方 llama.app Windows build 11574 的真實探測結果供 ATYL 回歸研究（見 [實機回歸證據](windows-build11574-fit-reference.md)），但仍應在每次更新引擎後重新執行；這份手動 Smoke Test 尚未取代後續的模型載入相容性測試。
