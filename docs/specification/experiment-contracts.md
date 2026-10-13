# ATYL Profile and Result Contract v1.1

These are **research-stage JSON Schema contracts**. They define what must be captured *before* running a future tuner; no model was executed to produce the examples.

## Workload profile

Required: pinned `engine` binary, subcommand where applicable, and commit, GGUF model SHA-256, concrete workload (prompt/generation/context/concurrency), objective, quality/service constraints, user-locked CLI values and a candidate proposal.

All parameters use a **canonical CLI name** and preserve the original CLI value type; the flag inventory does not yet verify individual value types. Lists are allowed for future multi-device arguments. Validation does not claim the CLI can accept every typed value.

**User locks cannot be overridden.** The validation tool rejects keys appearing in both `user_locked_parameters` and `candidate_parameters`. Context and concurrency minima must be respected. This cross-field policy is not encoded fully by JSON Schema alone.

## 版本識別 v1.1（與既有實驗檔不向下相容）

現在 `schema_version` 是 `1.1.0`。此前的示範檔 `1.0.0` 只是研究樣例，沒有自動改寫舊資料。需要先按照以下分離方式遷移，而不是把研究來源 Commit 當作實際執行檔的 Build。

- `registry_upstream_commit`（頂層、40 碼 SHA）：固定 CLI Inventory 的上游文件版本；只能說明文件來源。
- `engine.build_commit`：**本機安裝 Build 的 Commit**，尚未辨識到時必須是 `null`。
- `engine.binary_sha256`：真正探測的 exe 檔案指紋；未探測時為 `null`。
- `engine.version_output`：由 `llama version` 回報的版本文字；未探測時為 `null`。
- `engine.backend`：已證實的後端；未完成本機調查時為 `null`，不可因官方安裝器可能選擇 CUDA 而假設已選中。
- `run-record` 分別記錄 `engine_build_commit`、`engine_binary_sha256`、`registry_upstream_commit`、`entrypoint`，可額外記錄 `identity_cache_key`。

Registry 版本與真實引擎版本不能相互推論。CI 使用的 Profile／Result 樣例都故意標記為 `not_started`，所以不存在虛構的執行檔雜湊或 Backend。

## Engine entrypoint in profiles

Prefer `engine.binary: "llama"`, `engine.command: "bench"` or `"serve"`, and `engine.installation_source: "llama.app"`. The registry target is inferred as `llama-bench` or `llama-server` *after resolving the subcommand*. Legacy profiles using `llama-bench` or `llama-server` may omit `engine.command`; conflicting explicit commands are rejected. The workload `mode` must match the effective command. This logic now uses `tools/engine_adapter.py` (the same central routing rules used by the probe).

Do not hard-code WindowsApps into the profile. The eventual process record's `configuration.launched_argv` must include the real executable path **and subcommand**. Updating `llama.exe` changes the binary fingerprint and invalidates old capability measurements.

## Run record

- `configuration.requested`: intended startup parameters (including immutable locks).
- `configuration.fitted`: the fitting tool's proposed configuration or `null` if not observed.
- `configuration.effective`: independently confirmed runtime values or `null` if unverified.
- `configuration.launched_argv`: the exact argument vector passed to the process (an empty list for an unstarted illustrative run).
- `status` distinguishes success, failure, timeout, cancellation and no run.

Do not mistake requested CLI arguments for effective values. In particular, `--fit` may transform placement. Never infer `effective` solely from requested or `launched_argv`.

## Run checks

    python -m pip install -r requirements-dev.txt
    python tools/validate_contracts.py profile examples/workload-profile.example.json
    python tools/validate_contracts.py result examples/run-record.example.json
    python -m unittest discover -s tests -v

Example model SHA-256 is **64 zeros and is synthetic**. No benchmark, real hardware measurement, quality check or GPU allocation is represented by this fixture.

## Security and reproducibility

Do not persist authentication tokens, API keys or raw user prompts inside argv or a sharable result. Before exporting, redact local file paths and privacy-sensitive workload contents. The schemas cannot by themselves enforce secret redaction.

Next milestone: validate CLI types and capabilities against the installed binary; then record actual fit output and effective configuration from controlled benchmark runs.

## Optional speculative search/result extensions (backwards compatible)

The shared [`speculative-search.schema.json`](../../schemas/speculative-search.schema.json) defines `search` and `observation` shapes. Profiles may include `speculative_search` with explicit `modes` (including `off` baseline), bounded `draft_n_max_candidates`, `max_candidates`, external draft artifacts and device/layer placements. Run records may include `speculative_observation`, keeping requested N and independently observed effective N separate, together with nullable acceptance, latency, throughput and memory readings.

These optional extensions deliberately retain the existing parent `schema_version = 1.1.0`; historical profiles/results without them remain valid. The local validator resolves both shared references without contacting GitHub; it also rejects conflicting user CLI locks, missing external artifacts and impossible accepted-token counts. `not_started` examples contain only `null` measurements—**no fabricated benchmark evidence**.

`tools/plan_speculative_search.py` is a schema-validated **preview only**. It does not transform a speculative candidate into a runnable CLI command. For the phased execution plan see [tuning roadmap](tuning-roadmap.md).

For non-disabled speculative modes, use a **server workload** profile with `engine.command: serve` (the current framework does not claim `llama bench` exercises full speculative generation). See `examples/workload-profile.speculative.example.json`; the original ordinary benchmark example remains free of speculative settings to demonstrate backward compatibility.

## Pre-Feasibility integration guarantee

`tools/plan_speculative_search.py --profile ... --model-observation ... --runtime-snapshot ...` now passes the **entire Profile** through validation, rather than synthesizing empty locks and a default workload. All pinned speculative aliases (including `-ngld`, `--n-gpu-layers-draft`, `--model-draft` and option families) are checked against user-locked/candidate parameters. Profile `engine.binary_sha256` must match observed binary SHA when specified; recorded `--fit-plan` full-GGUF hash evidence is additionally checked against the Profile model SHA and observation source filename/size. Missing identity evidence blocks enabled MTP, not the non-executable baseline. This is not an actual disk or runtime proof.

The new optional Run Record `candidate_id` contains the 64-character identity of a planned candidate, **not** the per-execution `run_id`. For any completed speculative measurement it is required; old not-started fixtures remain valid. The `candidate_id` remains stable for the same complete planning inputs; actual future execution must revalidate files and runtime before starting.
