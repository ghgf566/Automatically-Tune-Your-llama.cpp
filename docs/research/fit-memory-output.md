# Fit-params memory breakdown — Windows build 11574

User-provided real `fit-daily.json` from an opt-in estimate with official `llama.exe` 0.6.0-dev build 11574, Qwen3.8-27B IQ4_XS, device `Vulkan0`, context 4096 and 18 GPU layers returned exit code 0.

The original two-row estimate output (de-identified; no local paths) was:

```text
Vulkan0 4136 101 525
Host 9111 304 44
```

The order and units are confirmed by the **pinned** upstream [common/fit.cpp::common_fit_print](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/common/fit.cpp): model, context, compute (MiB). No model was actually loaded for inference.

| Device | Model MiB | Context MiB | Compute MiB | Estimated sum MiB |
| --- | ---: | ---: | ---: | ---: |
| Vulkan0 | 4136 | 101 | 525 | 4762 |
| Host | 9111 | 304 | 44 | 9459 |
| **Combined** | **13247** | **405** | **569** | **14221** |

The user-declared `daily` scenario (Wallpaper Engine running) reported pre-estimate **7497 MiB free on Vulkan0** and **22634 MiB available system RAM**. After the estimate those values were 7497 and 22601 MiB. The provisional **2735 MiB** difference between reported GPU free memory and estimated GPU demand is not a guaranteed headroom reserve. Neither these estimates nor the before/after snapshots measure peak memory or model-load success.

## New structured parser

New v0.3.0 fit estimate records may contain `estimate.parsed_numeric_memory` with `devices[]` and `host`, each recording `model_mib`, `context_mib`, `compute_mib`, `estimated_total_mib`. The aggregate `estimated_total_mib` is the sum of the rows, not physical observed peak memory.

To convert your already saved **v0.2.0** file without rereading GGUF or executing llama again:

```powershell
git pull
python tools/parse_fit_record.py --input fit-daily.json --output fit-daily-parsed.json
```

The old original file is not overwritten. The archived v0.2.0 capture always set `output_truncated_or_unknown=true` conservatively, even for short output. The offline parser only accepts an exact complete, expected device/Host table. Modern v0.3.0 executions will not parse a known truncated output.

The parser fails closed on malformed, duplicated, partial or unexpected device rows and nonzero-exit output. If upstream changes the format, it reports unknown rather than inventing numbers. The tests contain only the two numeric lines, never the original full JSON or local file paths.

## Limitations and next work

`--fit off --fit-print on` estimates one fixed candidate, not an optimization. Runtime parameters such as context, GPU layers, KV cache quantization, batch/microbatch sizes and speculative or multimodal models can change memory needs. A model-load feasibility runner, peak sampling, and a benchmark suite are separate follow-on work.
