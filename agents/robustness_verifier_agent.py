from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from synthetic_openfoam_pipeline.config import PipelineConfig
from synthetic_openfoam_pipeline.schemas import RunnerMode, ScoreResult, TaskStatus, ValidationResult
from synthetic_openfoam_pipeline.utils.subprocesses import CommandResult, run_command


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

NAIVE_CLONE_SUFFIX = "-naive-check"


class RobustnessVerifierAgent:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def validate_task_dir(self, task_dir: Path) -> ValidationResult:
        task_dir = task_dir.resolve()
        task_id = task_dir.name
        issues = [rel for rel in REQUIRED_TASK_FILES if not (task_dir / rel).exists()]
        if issues:
            oracle = ScoreResult(0.0, False, {})
            naive = ScoreResult(0.0, False, {"skipped": "required task files missing"})
        else:
            oracle = self._run_ground_truth(task_dir)
            naive = self._run_naive_baseline(task_dir)
            if oracle.score < self.config.oracle_threshold:
                issues.append(f"oracle score {oracle.score} below threshold {self.config.oracle_threshold}")
            elif not oracle.passed:
                issues.append("oracle did not pass all hidden variants")
            if not naive.details.get("score_found", False):
                issues.append("naive baseline score could not be determined")
            elif naive.score > self.config.naive_threshold:
                issues.append(f"naive score {naive.score} exceeds threshold {self.config.naive_threshold}")
        hidden_passed = oracle.passed
        status = TaskStatus.CANDIDATE if not issues else TaskStatus.REJECTED
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
            if path.is_dir() and path.name.startswith("openfoam-") and not path.name.endswith(NAIVE_CLONE_SUFFIX)
        ]

    def _harness_command(self, problem_name: str, extra: list[str] | None = None) -> list[str]:
        command = [
            "uv",
            "run",
            "lbx-rl-harness",
            "run",
            "--runtime",
            "ground-truth",
            "--problem-dir",
            f"problems/{problem_name}",
        ]
        return command + (extra or [])

    def _persist_logs(self, task_id: str, label: str, run: CommandResult, output: str) -> dict[str, str]:
        log_dir = self.config.reports_root / task_id / label
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "harness_stdout.log"
        stderr_path = log_dir / "harness_stderr.log"
        combined_path = log_dir / "harness_combined.log"
        stdout_path.write_text(run.stdout)
        stderr_path.write_text(run.stderr)
        combined_path.write_text(output)
        return {
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "combined_log": str(combined_path),
        }

    def _run_ground_truth(self, task_dir: Path) -> ScoreResult:
        command = self._harness_command(task_dir.name)
        timeout = max(int(self.config.harness_timeout_sec), 1800)
        run = run_command(command, cwd=self.config.template_root, timeout=timeout)
        output = f"{run.stdout}\n{run.stderr}"
        score = self._score_from_output(output)
        log_paths = self._persist_logs(task_dir.name, "oracle", run, output)
        return ScoreResult(
            score,
            run.ok and score >= self.config.oracle_threshold,
            {
                "command": " ".join(command),
                "stdout": run.stdout,
                "stderr": run.stderr,
                "returncode": run.returncode,
                **log_paths,
            },
        )

    def _run_naive_baseline(self, task_dir: Path) -> ScoreResult:
        """Execute baselines/naive.sh through the same harness ground-truth runtime.

        The harness 'solution' runtime always executes solution/solve.sh, so the
        task is cloned to a scratch sibling problem dir whose solve.sh is the
        naive baseline script; a sibling keeps the Docker build context and
        PROBLEM_DIR build-arg valid.
        """
        clone_dir = task_dir.parent / f"{task_dir.name}{NAIVE_CLONE_SUFFIX}"
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        shutil.copytree(task_dir, clone_dir, ignore=shutil.ignore_patterns(".alignerr"))
        (clone_dir / "solution" / "solve.sh").write_text((clone_dir / "baselines" / "naive.sh").read_text())

        run_dir_base = (self.config.reports_root / task_dir.name / "naive" / "harness-runs").resolve()
        if run_dir_base.exists():
            shutil.rmtree(run_dir_base)
        run_dir_base.mkdir(parents=True, exist_ok=True)

        command = self._harness_command(clone_dir.name, ["--run-dir", str(run_dir_base)])
        timeout = max(int(self.config.harness_timeout_sec), 1800)
        try:
            run = run_command(command, cwd=self.config.template_root, timeout=timeout)
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)
        output = f"{run.stdout}\n{run.stderr}"
        score, score_source = self._naive_score(run_dir_base, output)
        log_paths = self._persist_logs(task_dir.name, "naive", run, output)
        return ScoreResult(
            score if score is not None else 0.0,
            score is not None and score <= self.config.naive_threshold,
            {
                "command": " ".join(command),
                "stdout": run.stdout,
                "stderr": run.stderr,
                "returncode": run.returncode,
                "score_found": score is not None,
                "score_source": score_source,
                "threshold": self.config.naive_threshold,
                **log_paths,
            },
        )

    def _naive_score(self, run_dir_base: Path, output: str) -> tuple[float | None, str]:
        # Primary: the grader's reward.json from the harness run dir. The naive
        # baseline is expected to score < 1.0, which makes the ground-truth
        # runtime exit non-zero after grading, so the exit code is not a signal.
        rewards = sorted(run_dir_base.glob("**/reward.json"), key=lambda path: path.stat().st_mtime)
        for reward_path in reversed(rewards):
            try:
                return float(json.loads(reward_path.read_text())["score"]), "reward.json"
            except (ValueError, KeyError, TypeError, OSError):
                continue
        explicit = re.findall(r"(?im)^\s*score:\s*([01](?:\.\d+)?)\s*$", output)
        if explicit:
            return max(float(match) for match in explicit), "harness-output"
        gate = re.findall(r"must\s+score\s+1\.0,\s*got\s+([01](?:\.\d+)?)", output)
        if gate:
            return float(gate[-1]), "gate-message"
        return None, "missing"

    def _score_from_output(self, output: str) -> float:
        matches = re.findall(r"(?im)^\s*score:\s*([01](?:\.\d+)?)\s*$", output)
        return max((float(match) for match in matches), default=0.0)
