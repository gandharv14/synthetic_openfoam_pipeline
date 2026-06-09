from __future__ import annotations

from pathlib import Path
from typing import Protocol

from synthetic_openfoam_pipeline.schemas import TargetAttemptResult


class TargetModel(Protocol):
    name: str

    def attempt_task(self, task_dir: Path, attempt: int, reports_dir: Path) -> TargetAttemptResult:
        ...

