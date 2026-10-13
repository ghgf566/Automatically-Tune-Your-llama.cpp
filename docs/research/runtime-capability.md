# Runtime CLI Capability Probe v0.1

This is the first read-only step beyond the pinned llama.cpp documentation inventory. **It does not benchmark, load a model, start a server, install a binary or tune configuration.**

## Usage (Windows PowerShell)

From the ATYL repository root, with a **trusted** llama.cpp binary:

```powershell
python tools/probe_runtime.py --binary "D:\llama.cpp\llama-server.exe" --output "my-server-capability.json"
python tools/probe_runtime.py --binary "D:\llama.cpp\llama-bench.exe" --include-devices --output "my-bench-capability.json"
```

Without --include-devices, only --version and --help are invoked. --list-devices is opt-in because device discovery can initialize GPU drivers. Each command has a bounded timeout (default 10 seconds). Execution is always via an argv array, **never shell interpolation**. No model path is passed to the executable.

## Output meaning

- Records the executable basename and SHA-256, upstream registry commit, and observation timestamp; it intentionally omits the executable's absolute path.
- Lists registry flags observed in the binary's --help, documented flags not observed, and flags listed in help but absent from ATYL's pinned registry.
- Version and optionally device output are truncated to 4096 characters, so private data in paths or tool output should still be reviewed before sharing snapshots.
- Unknown, removed or missing options are **not automatically declared unsupported**. Parsing the help text is intentionally conservative and can miss unusual layouts.
- A nonzero/timeout status is recorded rather than misclassified as a lack of feature support.
- Only directly named llama-server(.exe) or llama-bench(.exe) binaries are accepted in v0.1. Unified llama subcommands, wrappers and arbitrary renames need separate adapters.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Runtime probe tests use mocks and a synthetic dummy file. GitHub Actions never runs untrusted llama.cpp binaries. Actual hardware capabilities still require an explicit local run.

## Next

Compare differences between binary help and pinned documentation, confirm accepted values, collect build/backend metadata, and only then consider safe benchmark candidates.
