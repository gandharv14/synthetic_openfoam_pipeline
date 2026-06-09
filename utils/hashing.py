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
    short_names = {
        "boundary_condition_repair": "bc-repair",
        "postprocessing_instrumentation": "postproc",
        "solver_convergence_repair": "converge",
        "mesh_quality_repair": "mesh",
        "source_term_calibration": "source",
    }
    digest = stable_hash({"seed": seed, "family": family, "case_kind": case_kind, "index": index}, 8)
    return f"openfoam-{short_names.get(family, family.replace('_', '-'))}-{digest}"

