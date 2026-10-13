"""Fail-closed structural contract for runtime capability snapshots.

Complete CLI snapshots and reduced, synthetic test snapshots share a validated
minimum shape; unknown future fields are preserved.
"""
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas/runtime-capability.schema.json").read_text("utf-8"))
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA)


class SnapshotValidationError(ValueError):
    pass


def check_snapshot(document):
    try:
        VALIDATOR.validate(document)
    except ValidationError as error:
        field = ".".join(str(x) for x in error.absolute_path) or "root"
        raise SnapshotValidationError("invalid runtime snapshot field: " + field) from error
    binary = document["binary"]["sha256"]
    identity = document["identity"]["binary_sha256"]
    if binary != identity:
        raise SnapshotValidationError("snapshot binary SHA disagrees with identity")
    inventory = document.get("device_inventory")
    if inventory:
        status, devices = inventory["status"], inventory["devices"]
        if status == "observed" and not isinstance(devices, list):
            raise SnapshotValidationError("observed devices requires device array")
        if status == "unknown" and devices is not None:
            raise SnapshotValidationError("unknown device inventory must be null")
        if status == "none" and devices != []:
            raise SnapshotValidationError("none device inventory must be empty")
        for item in devices or []:
            if (item.get("total_memory_mib") is not None
                    and item.get("free_memory_mib") is not None
                    and item["free_memory_mib"] > item["total_memory_mib"]):
                raise SnapshotValidationError("device reports more free than total memory")
    return document
