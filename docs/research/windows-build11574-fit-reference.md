# Windows build 11574 — Regression Evidence and Fit Subset

ATYL received three local observation JSON files for **official `llama.exe` `0.6.0-dev` build `11574`**, with a reported abbreviated engine commit `8bb3cf0d2` and one SHA-256 common to all three probes. These were collected with `serve` (including Vulkan device listing), `bench`, and `fit-params`.

**Security/privacy:** the original observation JSON files are not committed to this repository. The regression fixture in `tests/fixtures/windows-cli-evidence.json` retains only a short build-info string, a small handful of observed flag names, and a synthetic device-output line. No local user names, paths, model contents, or original executable fingerprints are embedded.

## Verified from supplied snapshot, not benchmark results

- `serve --list-devices` completed successfully and reported one `Vulkan0` entry on an NVIDIA GTX 1070 Ti. Device memory is a GGML momentary report and does not establish backend/kernel model compatibility.
- `bench --help` showed `--offline` and `--version`, which the pinned bench README-derived inventory does not cover; they are not silently added to its 45 entries.
- `bench --help` did not show `--rpc`. The pinned upstream implementation at [tools/llama-bench/llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/tools/llama-bench/llama-bench.cpp) prints this help option only if `llama_supports_rpc()` is true. This is classified as *conditional in upstream*, not guaranteed unsupported.
- The prior `serve` snapshot included `--spec-ngram-` among unregistered flags. It ends in a hyphen and is not a valid complete CLI flag name; the parser now discards trailing-hyphen tokens. The full original `serve --help` text was not captured, so the precise source line has not been established.
- `fit-params --help` exposed `--fit-print` and fitting-related options; only the curated subset below has documented upstream evidence.

## Fit capability reference

The research baseline is [data/fit-capability-reference.json](../../data/fit-capability-reference.json), a **13-flag curated subset**, pinned to `77c4540ce4667f4577753fe034a3a8042f510b8f`. It is not a complete inventory of the 197 observed fit Help tokens (which include aliases). This subset keeps flag meaning distinct from observed presence and eventual model compatibility.

Read upstream [fit-params README](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/tools/fit-params/README.md), [fit-params source](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/tools/fit-params/fit-params.cpp), and [common arg parser](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/common/arg.cpp).

**Important:** `--fit-print on` prints estimated device/model/context/compute memory, rather than the default fitted CLI arguments. In the default mode, the utility may print fitted arguments such as `-c`, `-ngl`, `-ts`, and `-ot`. ATYL does **not** run either mode yet; Help-only probing continues to avoid model loading and unwanted work.

## Runtime outputs added

- `identity.engine_build_commit` now records a reported abbreviated commit if and only if a single unambiguous official `version:` build-info line matches; `engine_build_commit_form` states `abbreviated` or `full`, and `engine_build_number` records the reported integer. This is not copied from `registry_upstream_commit`.
- `diagnostics.missing_documented[]` and `diagnostics.unregistered_observed[]` carry **advisory** classifications. Missing entries retain support `not_confirmed`.
- `fit_reference.status = curated_reference_checked` lists only curated flags observed/missing in Help. The original full Registry comparison remains `null` for `fit-params`.

## When rerunning

From a trusted `llama.exe` installed through llama.app, use:

```powershell
python tools/probe_runtime.py --command serve --include-devices --output devices-capability-v2.json
python tools/probe_runtime.py --command bench --output bench-capability-v2.json
python tools/probe_runtime.py --command fit-params --output fit-capability-v2.json
```

This regenerates evidence under the repaired parser. It will not perform a benchmark, fit a model, or alter the installed runtime. GitHub CI uses synthetic/mock data and does not replace this local verification.

## First real memory estimate

An opt-in daily scenario estimate for Qwen3.8-27B IQ4_XS, `Vulkan0`, context 4096 and 18 offloaded layers returned rows `Vulkan0 4136 101 525` and `Host 9111 304 44` (model/context/compute MiB); exit code was 0. There was **no inference or model load test**. The reduced rows now form [Fit parser regression evidence](fit-memory-output.md).
