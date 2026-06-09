from __future__ import annotations

import re
from pathlib import Path

from synthetic_openfoam_pipeline.config import PipelineConfig
from synthetic_openfoam_pipeline.schemas import RunnerMode, ScoreResult, TaskStatus, ValidationResult
from synthetic_openfoam_pipeline.utils.subprocesses import run_command


REQUIRED_TASK_FILES = [
    "task.toml",
    "metadata.json",
    "instruction.md",
    "README.md",
    "environment/Dockerfile",
    "data/starter_case.tar.gz",
    "data/task_manifest.json",
    "scorer/__init__.py",
    "scorer/compute_score.py",
    "scorer/openfoam_grader_lib.py",
    "scorer/data/hidden_variants.json",
    "scorer/data/expected_metrics.json",
    "scorer/data/forbidden_edits.json",
    "scorer/data/pristine_case.tar.gz",
    "solution/solve.sh",
    "solution/solution_notes.md",
    "solution/render.sh",
    "baselines/naive.sh",
    "tests/test.sh",
]


class RobustnessVerifierAgent:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def validate_task_dir(self, task_dir: Path) -> ValidationResult:
        task_dir = task_dir.resolve()
        task_id = task_dir.name
        issues = [rel for rel in REQUIRED_TASK_FILES if not (task_dir / rel).exists()]
        oracle = self._run_ground_truth(task_dir) if not issues else ScoreResult(0.0, False, {})
        naive = ScoreResult(
            0.0,
            True,
            {"skipped": "real OpenFOAM validation is performed by the in-container ground-truth harness"},
        )
        hidden_passed = oracle.passed
        if issues:
            status = TaskStatus.REJECTED
        elif oracle.score < self.config.oracle_threshold:
            issues.append(f"oracle score {oracle.score} below threshold {self.config.oracle_threshold}")
            status = TaskStatus.REJECTED
        elif not hidden_passed:
            issues.append("oracle did not pass all hidden variants")
            status = TaskStatus.REJECTED
        else:
            status = TaskStatus.CANDIDATE
        return ValidationResult(
            task_id=task_id,
            status=status,
            oracle=oracle,
            naive=naive,
            hidden_passed=hidden_passed,
            runner_mode=RunnerMode.REAL,
            issues=issues,
        )

    def validate_many(self, tasks_root: Path) -> list[ValidationResult]:
        return [
            self.validate_task_dir(path)
            for path in sorted(tasks_root.iterdir())
            if path.is_dir() and path.name.startswith("openfoam-")
        ]

    def _run_ground_truth(self, task_dir: Path) -> ScoreResult:
        problem_dir = f"problems/{task_dir.name}"
        command = [
            "uv",
            "run",
            "lbx-rl-harness",
            "run",
            "--runtime",
            "ground-truth",
            "--problem-dir",
            problem_dir,
        ]
        timeout = max(int(self.config.harness_timeout_sec), 1800)
        run = run_command(command, cwd=self.config.template_root, timeout=timeout)
        output = f"{run.stdout}\n{run.stderr}"
        score = self._score_from_output(output)
        return ScoreResult(
            score,
            run.ok and score >= self.config.oracle_threshold,
            {
                "command": " ".join(command),
                "stdout": run.stdout,
                "stderr": run.stderr,
                "returncode": run.returncode,
            },
        )

    def _score_from_output(self, output: str) -> float:
        matches = re.findall(r"(?im)^\s*score:\s*([01](?:\.\d+)?)\s*$", output)
        return max((float(match) for match in matches), default=0.0)

