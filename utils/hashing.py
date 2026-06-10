from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(data: Any, length: int = 10) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)[:length]


def task_id_for(seed: int, family: str, case_kind: str, index: int) -> str:
    # The public ID deliberately omits the family so the title cannot be
    # pattern-matched to the repair; the digest still keys on family/seed/
    # case-kind/index so IDs stay deterministic and unique.
    digest = stable_hash({"seed": seed, "family": family, "case_kind": case_kind, "index": index}, 12)
    return f"openfoam-case-{digest}"

