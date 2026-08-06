"""Atomic file writes — never leave a half-written file at the target path.

Every persistent write in the pipeline (index, sidecars, manifests, metadata
records) goes through these helpers. The pattern is the same: write a sibling
unique ``*.tmp`` staging file first, then ``os.replace`` it onto the target.
``os.replace`` is POSIX-atomic on the same filesystem, so a
SIGINT/SIGKILL/OOM between writes either leaves the prior version intact or
installs the new one — never a truncated mix of the two.

The staging name is unique per call (``<name>.<pid>.<rand>.tmp``) so two
concurrent writers to the same target can never interleave on a shared
staging file and install a partial mix — each writer stages privately and
the last ``os.replace`` wins whole.

The companion R5 integrity sidecar detects a corrupted index AFTER the fact;
these helpers PREVENT the corruption in the first place.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def _staging_path(path: Path) -> Path:
    """Unique sibling staging path for ``path`` (same dir → same filesystem,
    so ``os.replace`` stays atomic). Keeps the ``.tmp`` suffix convention."""
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``path`` atomically.

    Writes to a unique sibling ``*.tmp`` first, then ``os.replace`` it onto
    the final target. A crash mid-write leaves the prior file (if any)
    intact; concurrent writers to the same target never share a staging file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _staging_path(path)
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(tmp, path)
    except BaseException:
        # Best-effort cleanup of the private staging file; the target is
        # untouched either way.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Binary counterpart to ``_atomic_write_text``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _staging_path(path)
    try:
        tmp.write_bytes(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
