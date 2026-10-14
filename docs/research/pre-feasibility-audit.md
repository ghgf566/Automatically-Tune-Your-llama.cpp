# Final Pre-Feasibility Repository Audit

**Status:** Source architecture and research/preview contracts are ready to begin implementing a *separate opt-in* Feasibility Runner. This is NOT approval to execute unverified MTP pairs or proof that real model loads already work.

## Scope reviewed

- [x] CLI Registry pinned to its documented upstream revision (302 CLI rows, 32 tuning-knowledge entries), command routing, engine version/Help/device snapshots.
- [x] Bounded GGUF v2/v3 header/metadata inspector, shard declaration, NextN/MTP field, per-device metadata and explicit unknown semantics.
- [x] Single-candidate `llama fit-params` estimation (off by default; `--execute` required), GGML MiB parser, old JSON conversion, before/after RAM/GPU headroom.
- [x] Model fingerprinting: low-I/O prefix + file stat, opt-in full-file SHA256; binary SHA bound to runtime snapshots.
- [x] Workload Profile → speculative search → preview Candidate ID → optional speculative Result with requested vs observed Draft N.
- [x] Safe artifact writes, CLI exit codes, sandbox-free child-process invocation, timeout and best-effort child-tree cleanup.
- [x] CI test matrix: Windows/Linux × Python 3.11/3.13, unit tests, static error lint, Schema/example validation, Registry consistency, statement coverage reporting.

## Confirmed corrections during this audit

1. **Metadata-to-model evidence binding**: earlier previews could accept a recorded full-file Fit hash together with an unrelated GGUF Metadata JSON of the same filename/size. New `metadata_sha256` hashes the exact inspected GGUF prefix using the same open file handle. If `--fit-plan` is provided, the Planner cross-checks the prefix digest, byte span, filename, size, architecture, full-file hash record and engine SHA. Absent historical digest is not silently assumed valid.
2. **Run Record success semantics**: `effective_draft_n_max` cannot exceed the configured requested maximum; a completed speculative measurement needs requested Draft N; a successful on/off speculative observation needs a completed measurement and recorded successful model-load evidence. Existing not-started synthetic examples remain valid.
3. **Cross-module synthetic evidence fixture**: a real byte-level synthetic GGUF is passed through Inspector and full-file Fingerprint, then matched by Profile-backed MTP Planner. A swapped prefix digest must fail. No real model or CUDA/Vulkan backend is used by this test.

## Preconditions for the first actual Feasibility Runner

1. **Explicit consent**: only an opt-in operation may start real `llama serve`/target+draft loads; preview and existing Fit have different authorization scopes.
2. **Trust-on-execution**: verify the current on-disk target GGUF, complete shard/bundle components, any mmproj and external Draft file, and the live binary/device identity again immediately before load. JSON digests alone are historical evidence, not authorization.
3. **Mode/value verification**: Help flag presence does not establish accepted `--spec-type` values, true embedded/external MTP compatibility, or supported `draft_n_max` limits. Verify actual running model/backend.
4. **Resource bounds**: memory reservation is provisional until loaded and sampled; add continuous peak sampling, concurrency, timeouts and explicit cancellation. Existing process-group/taskkill cleanup is best-effort, not an OS containment boundary for detached children.
5. **Correct outcomes**: classify unsupported mode, missing/invalid draft pair, model load failure, OOM, timeout, cancellation, successful load, and benchmark separately. Do not report throughput or acceptance-rate data from failed runs.
6. **Controlled comparison**: measure baseline without MTP before Draft N variations. Keep workload/context/sampling/target offload fixed first, then explore interactions if evidence permits.

## Historical results and limitations

- Existing Windows build 11574 `Vulkan0` GGML Fit rows are real **memory estimates**, not a successful target or Draft inference load.
- The GGUF prefix hash is not a full weight hash; `--full-model-sha256` deliberately reads the entire model only on explicit request.
- Existing old GGUF observation JSON without `metadata_sha256` is still readable, but **cannot be paired as verified Fit metadata**. Regenerate it using the current Inspector (`--overwrite` only when you intentionally replace an existing JSON).
- External Draft pairing, split-GGUF completeness, `mmproj`, backend-specific quality/throughput and real GPU load tests have **not** been completed.
- Future Windows service processes may need Job Objects if best-effort `/T /F` process-tree termination is insufficient; the current CI checks descendants spawned by ATYL, not arbitrary third-party services.

## PowerShell: update and run non-executable checks

```powershell
git pull
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python tools/validate_contracts.py profile examples/workload-profile.speculative.example.json
python tools/validate_contracts.py result examples/run-record.example.json
python tools/generate_cli_registry.py --check
```

This source audit and CI cannot replace a subsequent user's explicit, hardware-specific smoke test before enabling real-model loading in ATYL.
