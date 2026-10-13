#!/usr/bin/env python3
"""Bounded, read-only GGUF v2/v3 metadata inspection (never loads model weights).

Source format: llama.cpp ggml/include/gguf.h and gguf-py/gguf/gguf_reader.py,
pinned at 77c4540ce4667f4577753fe034a3a8042f510b8f.
"""
import argparse
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path

MAGIC = b"GGUF"
VERSIONS = {2, 3}
TYPE_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
}
TYPE_STRING, TYPE_ARRAY = 8, 9
MAX_HEADER_BYTES = 256 * 1024 * 1024
MAX_KEYS = 100_000
MAX_TENSORS = 1_000_000
MAX_ARRAY_ELEMENTS = 2_000_000
MAX_ITERATIONS = 3_000_000
MAX_KEY_BYTES = 4096
MAX_CAPTURE_BYTES = 8192
MAX_ALIGNMENT = 4096

GENERAL_KEYS = {
    "general.architecture", "general.name", "general.file_type",
    "general.quantization_version", "general.alignment",
    "split.count", "split.no", "split.tensors.count",
}
ARCH_SUFFIXES = {
    ".block_count", ".embedding_length", ".attention.head_count",
    ".expert_count", ".expert_used_count", ".feed_forward_length",
    ".context_length",
    # Upstream gguf-py Keys.Attention and Keys.Rope at the pinned commit.
    # Capture fields literally; do not infer KV cache size from incomplete keys.
    ".attention.head_count_kv", ".attention.key_length",
    ".attention.value_length", ".attention.sliding_window",
    ".attention.key_length_mla", ".attention.value_length_mla",
    ".attention.key_length_swa", ".attention.value_length_swa",
    ".attention.kv_lora_rank", ".attention.kv_lora_rank_swa",
    ".rope.dimension_count", ".rope.dimension_count_swa",
    ".rope.freq_base", ".rope.scaling.type", ".rope.scaling.factor",
    ".rope.scaling.original_context_length",
    ".expert_feed_forward_length", ".expert_shared_count",
}


class GGUFMetadataError(ValueError):
    pass


class BoundedReader:
    def __init__(self, stream, size, limit=MAX_HEADER_BYTES):
        self.stream, self.size, self.limit = stream, size, limit
        self.pos = 0
        self.iterations = 0

    def advance(self, size, *, capture=False):
        if size < 0 or self.pos + size > self.size:
            raise GGUFMetadataError("truncated GGUF header or metadata")
        if self.pos + size > self.limit:
            raise GGUFMetadataError("GGUF metadata exceeds scan limit")
        start = self.pos
        self.pos += size
        if capture:
            result = self.stream.read(size)
            if len(result) != size:
                raise GGUFMetadataError("short GGUF read")
            return result
        self.stream.seek(size, 1)
        return None

    def read(self, size):
        return self.advance(size, capture=True)

    def skip(self, size):
        self.advance(size)

    def unpack(self, fmt):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def text(self, *, capture, length_limit=MAX_CAPTURE_BYTES):
        size = self.unpack("<Q")
        if capture:
            if size > length_limit:
                raise GGUFMetadataError("captured GGUF string exceeds size limit")
            raw = self.read(size)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise GGUFMetadataError("invalid UTF-8 GGUF string") from error
        self.skip(size)
        return None

    def value(self, kind, *, capture=False):
        if kind in TYPE_FMT:
            fmt = TYPE_FMT[kind]
            if capture:
                value = self.unpack(fmt)
                if kind == 7 and value not in (0, 1):
                    raise GGUFMetadataError("invalid GGUF boolean")
                return bool(value) if kind == 7 else value
            self.skip(struct.calcsize(fmt))
            return None
        if kind == TYPE_STRING:
            return self.text(capture=capture)
        if kind == TYPE_ARRAY:
            element_type = self.unpack("<I")
            count = self.unpack("<Q")
            if element_type == TYPE_ARRAY or element_type not in (*TYPE_FMT, TYPE_STRING):
                raise GGUFMetadataError("invalid GGUF array element type")
            if count > MAX_ARRAY_ELEMENTS or self.iterations + count > MAX_ITERATIONS:
                raise GGUFMetadataError("GGUF array exceeds element scan limit")
            self.iterations += count
            if element_type == TYPE_STRING:
                for _ in range(count):
                    self.text(capture=False)
            else:
                self.skip(struct.calcsize(TYPE_FMT[element_type]) * count)
            return None
        raise GGUFMetadataError("unsupported GGUF value type: " + str(kind))


def wanted_key(key):
    return key in GENERAL_KEYS or any(key.endswith(suffix) for suffix in ARCH_SUFFIXES)


def inspect(path):
    """Only observes on-disk metadata; not a model validator or inference test."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise GGUFMetadataError("GGUF path must be an existing regular file")
    size = path.stat().st_size
    with path.open("rb") as file:
        reader = BoundedReader(file, size)
        if reader.read(4) != MAGIC:
            raise GGUFMetadataError("not a GGUF file")
        version = reader.unpack("<I")
        if version not in VERSIONS:
            raise GGUFMetadataError("unsupported GGUF version " + str(version))
        tensor_count, kv_count = reader.unpack("<Q"), reader.unpack("<Q")
        if tensor_count > MAX_TENSORS or kv_count > MAX_KEYS:
            raise GGUFMetadataError("implausible GGUF tensor or key count")
        metadata = {}
        seen_keys = set()
        for _ in range(kv_count):
            key = reader.text(capture=True, length_limit=MAX_KEY_BYTES)
            if not key or key in seen_keys:
                raise GGUFMetadataError("invalid or duplicate GGUF metadata key")
            seen_keys.add(key)
            kind = reader.unpack("<I")
            data = reader.value(kind, capture=wanted_key(key) and kind != TYPE_ARRAY)
            # Only retain explicitly whitelisted scalar fields, not paths,
            # free-form tokenizer data, private metadata or huge arrays.
            if wanted_key(key) and kind != TYPE_ARRAY:
                if isinstance(data, float) and not math.isfinite(data):
                    raise GGUFMetadataError("non-finite GGUF metadata value")
                metadata[key] = data

        alignment = metadata.get("general.alignment", 32)
        if (not isinstance(alignment, int) or isinstance(alignment, bool) or
                alignment < 1 or alignment > MAX_ALIGNMENT or
                alignment & (alignment - 1)):
            raise GGUFMetadataError("invalid general.alignment")
        if "general.architecture" in metadata and not isinstance(metadata["general.architecture"], str):
            raise GGUFMetadataError("general.architecture is not a string")
        if "general.file_type" in metadata and not isinstance(metadata["general.file_type"], int):
            raise GGUFMetadataError("general.file_type is not an integer")
        type_counts = Counter()
        for _ in range(tensor_count):
            reader.text(capture=False)  # Never retain tensor names.
            dims = reader.unpack("<I")
            if not 1 <= dims <= 4:
                raise GGUFMetadataError("invalid GGUF tensor dimensions")
            for _ in range(dims):
                if reader.unpack("<Q") == 0:
                    raise GGUFMetadataError("invalid zero-sized tensor dimension")
            ggml_type = reader.unpack("<I")
            reader.unpack("<Q")  # Tensor offset; weights remain unread.
            type_counts[str(ggml_type)] += 1
        metadata_end = reader.pos
        data_offset = ((metadata_end + alignment - 1) // alignment) * alignment
        if tensor_count and data_offset > size:
            raise GGUFMetadataError("tensor data offset exceeds file length")

    arch = metadata.get("general.architecture")
    arch_details = {key[len(arch) + 1:]: value for key, value in metadata.items()
                    if arch and key.startswith(arch + ".")}
    split_count = metadata.get("split.count")
    multi = isinstance(split_count, int) and split_count > 1
    return {
        "schema_version": "0.1.0",
        "kind": "gguf_metadata_observation",
        "source": {"filename": path.name, "file_size_bytes": size},
        "gguf_version": version,
        "tensor_count_in_this_file": tensor_count,
        "metadata_key_count": kv_count,
        "general": {
            "architecture": arch,
            "name": metadata.get("general.name"),
            "file_type_id": metadata.get("general.file_type"),
            "quantization_version": metadata.get("general.quantization_version"),
        },
        "architecture_metadata": arch_details,
        "tensor_type_id_counts": dict(sorted(type_counts.items(), key=lambda it: int(it[0]))),
        "split": {
            "part_index": metadata.get("split.no"),
            "part_count": split_count,
            "total_tensor_count_reported": metadata.get("split.tensors.count"),
            "multiple_parts_reported": multi,
            "all_parts_inspected": False,
        },
        "metadata_bytes_scanned": metadata_end,
        "validation": {
            "metadata_structure_parsed": True,
            "tensor_descriptors_parsed": True,
            "tensor_data_read": False,
            "tensor_payload_integrity_verified": False,
            "model_load_tested": False,
            "model_backend_compatibility": "not_tested",
        },
        "limitations": [
            "general.file_type is a numeric GGUF file-type label, not verified per-tensor quantization.",
            "This file may be only one of multiple GGUF shards or auxiliary projector files.",
            "Memory requirements, supported operations and actual model load remain unverified.",
            "Only whitelisted scalar metadata is retained; tensor data is never read.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Existing local GGUF")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    try:
        result = json.dumps(inspect(args.model), indent=2, ensure_ascii=False) + "\n"
        if args.output:
            if args.output.expanduser().resolve() == args.model.expanduser().resolve():
                raise GGUFMetadataError("output must not overwrite input GGUF")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result, encoding="utf-8")
            print("Created model metadata observation:", args.output)
        else:
            print(result, end="")
    except (OSError, GGUFMetadataError) as error:
        print("GGUF inspection failed:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
