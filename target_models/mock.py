from __future__ import annotations

import json
from pathlib import Path

from synthetic_openfoam_pipeline.schemas import TargetAttemptResult
from synthetic_openfoam_pipeline.utils.fs import write_json


class AlwaysFailTarget:
    name = "mock-fail"

    def attempt_task(self, task_dir: Path, attempt: int, reports_dir: Path) -> TargetAttemptResult:
        reports_dir.mkdir(parents=True, exist_ok=True)
        transcript = reports_dir / f"attempt_{attempt}.json"
        write_json(transcript, {"task_id": task_dir.name, "mock": "fail", "passed": False})
        return TargetAttemptResult(target=self.name, attempt=attempt, passed=False, score=0.0, transcript_path=str(transcript))


class AlwaysPassTarget:
    name = "mock-pass"

    def attempt_task(self, task_dir: Path, attempt: int, reports_dir: Path) -> TargetAttemptResult:
        reports_dir.mkdir(parents=True, exist_ok=True)
        transcript = reports_dir / f"attempt_{attempt}.json"
        transcript.write_text(json.dumps({"task_id": task_dir.name, "mock": "pass", "passed": True}, indent=2) + "\n")
        return TargetAttemptResult(target=self.name, attempt=attempt, passed=True, score=1.0, transcript_path=str(transcript))


class HeuristicTarget:
    name = "mock-heuristic"

    def attempt_task(self, task_dir: Path, attempt: int, reports_dir: Path) -> TargetAttemptResult:
        passed = "postproc" in task_dir.name
        reports_dir.mkdir(parents=True, exist_ok=True)
        transcript = reports_dir / f"attempt_{attempt}.json"
        write_json(transcript, {"task_id": task_dir.name, "mock": "heuristic", "passed": passed})
        return TargetAttemptResult(target=self.name, attempt=attempt, passed=passed, score=1.0 if passed else 0.0, transcript_path=str(transcript))


def mock_target(name: str):
    if name == "mock-fail":
        return AlwaysFailTarget()
    if name == "mock-pass":
        return AlwaysPassTarget()
    if name == "mock-heuristic":
        return HeuristicTarget()
    raise ValueError(f"Unknown mock target: {name}")

