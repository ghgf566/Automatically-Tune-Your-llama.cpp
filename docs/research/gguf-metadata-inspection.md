# GGUF Metadata Inspection v0.1 — model feasibility foundation

## Purpose

Before ATYL can estimate device memory or tune execution, it needs a reliable description of the **exact GGUF file selected by the user**. The first inspection stage is standalone Python and **never launches llama**, touches a server, downloads a model, reads tensor weight bytes, or makes benchmark claims.

Implementation: [`tools/inspect_gguf.py`](../../tools/inspect_gguf.py). Schema: [`schemas/gguf-observation.schema.json`](../../schemas/gguf-observation.schema.json). Python 3.9+ standard library; tests use `jsonschema` in the existing development requirements.

## Windows PowerShell

From the ATYL repository root, substitute your actual nested model path:

```powershell
python tools/inspect_gguf.py --model "D:\AIModels\Author\Repository\model.gguf" --output model-metadata.json
```

Or write JSON directly to the console:

```powershell
python tools/inspect_gguf.py --model "D:\AIModels\Author\Repository\model.gguf"
```

The scan reads GGUF header/KV information and walks tensor **descriptors** only. Tokenizer arrays are *skipped without materializing their contents*, and model weight blocks are not read. The inspector does not call the official `llama.exe` at all.

## Captured fields

| Output | Meaning |
| --- | --- |
| `gguf_version` | On-disk GGUF version, currently support v2 and v3 |
| `source.filename` / `source.file_size_bytes` | This file only, not necessarily all shards |
| `tensor_count_in_this_file` | Count from this GGUF header |
| `general.architecture` / `general.name` | Whitelisted GGUF strings, if supplied |
| `general.file_type_id` | Original **numeric** GGUF type ID; not a verified tensor quantization label |
| `general.quantization_version` | Format's quantization version, if supplied |
| `architecture_metadata` | Whitelisted numerical architecture keys such as `block_count`, `embedding_length`, `expert_count`, `expert_used_count`, `attention.head_count`, `attention.head_count_kv`, `attention.key_length`, `attention.value_length`, `rope.dimension_count` |
| `tensor_type_id_counts` | Count of reported tensor type IDs, *not validated tensor contents* |
| `split` | Part metadata when declared; **never assumes all companion files were inspected** |
| `validation` | Explicitly distinguishes parsed metadata from unverified weights and model/backend execution |

Unknown or absent metadata stays `null` or omitted; MoE capability is **not inferred** simply because an architecture string contains `moe`. Quantization format ID and tensor type IDs are preserved without an unchecked human-readable mapping.

## Safety boundaries

- The parser checks the `GGUF` magic, header version, dimensions, counts, UTF-8 strings, alignment and file boundaries. Unknown types, malformed or oversized entries fail with an error.
- To avoid unbounded metadata processing, it caps scanning at 256 MiB of header and descriptors, 100k KV pairs, 1 million tensors and 2 million elements in any one array. A legitimate model above the cap must be handled by a future more capable reader; the cap is intentional.
- No model files are modified. No full GGUF SHA-256 is calculated because hashing all model bytes would violate this stage's low-I/O purpose. File size alone does **not** uniquely identify a model.
- A successful output proves only the **parsed header and tensor-descriptor structure**. It does not verify tensor payload integrity, complete shards, runtime memory needs, GPU operator support, compatibility with an actual installed `llama.exe`, or inference quality.
- A multi-file GGUF may distribute tensors across shards; inspect each relevant part in a later aggregation phase. Multimodal `mmproj` can be another GGUF and must not be mistaken for the main language model.
- The JSON can include a model filename and general model name. Review it before sharing if these are private.

## Evidence and subsequent work

The layout and v2/v3 support are grounded in the upstream [GGUF header](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/ggml/include/gguf.h) and [GGUF Reader](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/gguf-py/gguf/gguf_reader.py), pinned to the registry's baseline commit. CI tests construct **synthetic GGUF binaries** and never execute a real model.

Next: `tools/fit_estimate.py` now previews one fixed-configuration memory-estimate command using a current binary-matched Fit snapshot and, for GPU usage, a device snapshot. Real GGUF reads by llama are gated behind `--execute`. See [controlled fit preview](fit-estimate-preview.md). This is not a benchmark or a tested fit, and file size/device free-memory comparisons are not a substitute for it.
