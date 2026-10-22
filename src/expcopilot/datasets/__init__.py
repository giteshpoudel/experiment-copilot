"""Public datasets loaded into the experiment store as real-data case studies.

Raw files live in data/raw/ (never committed) and are checked against a pinned SHA-256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class DatasetError(RuntimeError):
    pass


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, expected: str | None) -> str:
    """Raise if the file is missing; return its digest, raising on a mismatch."""
    if not path.exists():
        raise DatasetError(f"{path} not found")
    digest = sha256(path)
    if expected and digest != expected:
        raise DatasetError(f"{path} SHA-256 is {digest}, expected {expected}; re-download it")
    return digest
