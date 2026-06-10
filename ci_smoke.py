from __future__ import annotations

import subprocess
from pathlib import Path

from synthetic_openfoam_pipeline.config import DEFAULT_FAMILIES


SMOKE_FAMILIES = list(DEFAULT_FAMILIES)


def run_ground_truth_smoke(template_root: Path, *, families: list[str] | None = None, timeout: int = 1800) -> dict[str, int]:
    selected = families or SMOKE_FAMILIES
    results: dict[str, int] = {}
    for family in selected:
        matches = sorted((template_root / "problems").glob("openfoam-*"), key=lambda path: path.stat().st_mtime, reverse=True)
        task_dir = next((path for path in matches if _task_family(path) == family), None)
        if task_dir is None:
            results[family] = 127
            continue
        proc = subprocess.run(
            [
                "uv",
                "run",
                "lbx-rl-harness",
                "run",
                "--runtime",
                "ground-truth",
                "--problem-dir",
                f"problems/{task_dir.name}",
            ],
            cwd=template_root,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        results[family] = proc.returncode
    return results


def _task_family(task_dir: Path) -> str | None:
    import json

    report = task_dir / "generation_report.json"
    if not report.exists():
        return None
    return json.loads(report.read_text()).get("family")
