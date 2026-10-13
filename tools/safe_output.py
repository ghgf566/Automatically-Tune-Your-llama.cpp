"""Shared safe UTF-8 artifact writer; never replace an input or an existing output by default."""
import os
import tempfile
from pathlib import Path


class UnsafeOutputError(ValueError):
    pass


def write_text(destination, contents, *, protected=(), overwrite=False):
    """Write via sibling temporary file, then publish atomically on NTFS/ext4.

    By default, os.link provides exclusive create semantics: an existing target
    (including symlinks) is never overwritten. --overwrite uses os.replace.
    Resolve all protected paths to prevent aliases pointing at source inputs.
    """
    destination = Path(destination).expanduser().absolute()
    resolved = destination.resolve()
    sources = {Path(path).expanduser().resolve() for path in protected if path is not None}
    if resolved in sources:
        raise UnsafeOutputError("output path overlaps a protected input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=destination.parent,
                                         prefix=".atyl-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            # Re-check important paths immediately before replacement.
            if destination.resolve() in sources:
                raise UnsafeOutputError("output path now overlaps a protected input")
            os.replace(temporary, destination)
        else:
            # Fails without modifying an existing file even if another process
            # creates it concurrently. os.link is supported on NTFS and POSIX.
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise UnsafeOutputError(
                    "output exists; choose a new name or pass --overwrite") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
