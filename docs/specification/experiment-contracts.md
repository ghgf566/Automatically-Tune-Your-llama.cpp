# ATYL Profile and Result Contract v1

These are **research-stage JSON Schema contracts**. They define what must be captured *before* running a future tuner; no model was executed to produce the examples.

## Workload profile

Required: pinned `engine` binary and commit, GGUF model SHA-256, concrete workload (prompt/generation/context/concurrency), objective, quality/service constraints, user-locked CLI values and a candidate proposal.

All parameters use a **canonical CLI name** and preserve the original CLI value type; the flag inventory does not yet verify individual value types. Lists are allowed for future multi-device arguments. Validation does not claim the CLI can accept every typed value.

**User locks cannot be overridden.** The validation tool rejects keys appearing in both `user_locked_parameters` and `candidate_parameters`. Context and concurrency minima must be respected. This cross-field policy is not encoded fully by JSON Schema alone.

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
