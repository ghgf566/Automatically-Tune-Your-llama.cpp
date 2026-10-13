# ATYL variable architecture and delivery roadmap

**Goal:** Build a dependency-aware tuner, not a growing set of ad hoc UI toggles or unvalidated llama command strings. All modes must be gated by observed engine/model capabilities and backed by reproducible measurements. This roadmap is a plan, not a claim of implemented tuning.

## One shared planning contract

Research Profile (`workload-profile.schema.json`) → optional `speculative_search` (the shared schema's `search` definition) → candidate preview → model/load/memory validation → controlled benchmark → optional `speculative_observation` in Run Record (the shared schema's `observation` definition) → Pareto/reporting.

The search and observation definitions both live in **one** [`speculative-search.schema.json`](../../schemas/speculative-search.schema.json), resolved locally without network. New speculative families or result metrics are added there, rather than copied into independent CLI, GUI, benchmark and reporting schemas. The research Profile/Result versions remain `1.1.0` because these fields are optional; older files continue to validate.

## Variables and dependency groups

| Group | Fixed constraints / adjustable dimension | Prerequisites | Test evidence |
| --- | --- | --- | --- |
| Main model & auxiliary models | main GGUF fingerprint, external draft fingerprint, tokenizer/pairing, optional mmproj | all files explicitly selected and identified | metadata, later model load |
| Engine & backend | binary SHA, Help flags, mode-specific available values, GPUs/CPU | fresh matching snapshot | observed then tested, never assumed |
| Core placement | main GPU layers, tensor split, device, threads | actual backend and memory capacity | estimate then load |
| Context & memory | ctx, KV cache types, micro-batch, concurrency, memory reserve | workload minimum constraints | estimate/observed peaks |
| Speculative family | off / embedded MTP / external MTP / simple draft | mode-specific GGUF evidence, engine support, compatible inputs | load then benchmark |
| Speculative search | **draft_n_max**, draft placement, later acceptance thresholds | positive draft mode, bounded requested values and compatibility checks | effective vs requested N, drafted/accepted counts |
| Quality & policy | sampler, target model, context minimum, user locks | must not silently change | quality and workload contract |
| Environment | daily/reduced background, heat/power state, seed/repetitions | capture per-run state | statistically comparable repetitions |

Do not equate `nextn_predict_layers` with the upper limit for `draft_n_max`; actual MTP runtime implementation may place a different limit. Likewise do not assume `llama-bench` measures the full speculative server path. Mark missing data **unknown** and withhold unsupported candidates.

## Progress checkpoints

| Stage | Current state | Completion criterion |
| --- | --- | --- |
| Runtime / GGUF / Fit estimates / safe output / Windows CI | Implemented, partially real-Windows verified | maintain binary-linked snapshots and regression coverage |
| Typed speculative search + requested Draft N | **Schema and non-executable preview implemented** | enumerate explicit candidate budgets, preserve baseline, reject unsupported evidence |
| Target-draft pairing + multisplit/auxiliary identity | Not implemented | documented target/draft compatibility checks and model-bundle fingerprint |
| Speculative memory feasibility | Not implemented | model+draft memory under same target context and offload, opt-in actual load, report failures |
| Real speculative generation/serving benchmark | Not implemented | fixed prompts/sampling, accepted/drafted tokens, TTFT, p95, throughput, memory peaks, repetitions |
| Adaptive bounded optimization | Not implemented | compare off vs candidate, coarse-to-fine N search, joint placement refinement within cap, stop on regressions |
| UI / config/export | Not implemented | derive controls dynamically from plan/schema; expose source/evidence, never silently alter locked values |

## Suggested bounded search protocol

1. Create an **off** baseline under the same target model, build, workload, sampling and context constraints. Record actual backend/defaults.
2. For each eligible MTP or draft mode, probe a **small increasing set** (e.g. 1/2/4/8) of `draft_n_max`. These are requested maxima, not guaranteed accepted/generated token counts.
3. Verify whether the engine clamps or rejects the requested value; store **requested and effective** separately. If effective N is unobservable, report `null` and do not present it as exact.
4. Estimate and verify combined draft+target RAM/VRAM, then run warm-up and repeated measurements. GPU layers/draft device must be held fixed for the first comparison.
5. Refine N around measured promising values under a fixed budget. Only then study interactions with draft placement, main offload, Context and KV configuration; these are coupled variables.
6. Evaluate throughput **and** tail latency, TTFT, memory overhead, stability, sampling contract and acceptance counters (if available). Choose **off** if all speculative options lose; preserve Pareto trade-offs.
7. Only launch a real model/benchmark after explicit user consent and verified pairing, flags, limits and local input paths. Preview tools remain non-executable.

## Compatibility with earlier artifacts

The optional fields do **not** retrofit old Fit v0.3 JSON with hypothetical MTP measurements. A plain `fit-params` number cannot be taken as proof that an embedded or external MTP plan fits. Historic workload and result JSON without speculative fields continue to be valid; new runs may attach their planned spec search and observed result when those stages exist.
