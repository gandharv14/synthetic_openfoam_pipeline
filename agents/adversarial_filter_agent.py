from __future__ import annotations

from pathlib import Path

from synthetic_openfoam_pipeline.config import PipelineConfig
from synthetic_openfoam_pipeline.schemas import AdversarialResult, TaskStatus, ValidationResult
from synthetic_openfoam_pipeline.target_models.claude import ClaudeTarget
from synthetic_openfoam_pipeline.target_models.harness import TemplateHarnessTarget
from synthetic_openfoam_pipeline.target_models.mock import mock_target
from synthetic_openfoam_pipeline.utils.fs import repo_root
from synthetic_openfoam_pipeline.utils.fs import ensure_dir


class AdversarialFilterAgent:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run_for_task(self, task_dir: Path, validation: ValidationResult, target_name: str | None = None) -> AdversarialResult:
        target_name = target_name or self.config.target
        target = self._target(target_name)
        reports_dir = self.config.reports_root / task_dir.name / "adversarial" / target.name
        ensure_dir(reports_dir, allow_template_output=self.config.allow_template_output)
        attempts = [target.attempt_task(task_dir, i + 1, reports_dir) for i in range(self.config.attempts)]

        if validation.status == TaskStatus.REJECTED:
            return AdversarialResult(task_dir.name, target.name, TaskStatus.REJECTED, attempts, "validation failed before adversarial filtering")
        if any(not attempt.available for attempt in attempts):
            return AdversarialResult(task_dir.name, target.name, TaskStatus.NEEDS_CLAUDE_VALIDATION, attempts, "target unavailable")
        if any(attempt.passed for attempt in attempts):
            return AdversarialResult(task_dir.name, target.name, TaskStatus.REJECTED, attempts, "target solved at least one attempt")
        if any(attempt.status in {"error", "timeout"} for attempt in attempts):
            return AdversarialResult(task_dir.name, target.name, TaskStatus.NEEDS_CLAUDE_VALIDATION, attempts, "target or harness did not complete cleanly")
        if target.name in {"claude", "claude-harness"}:
            return AdversarialResult(task_dir.name, target.name, TaskStatus.ACCEPTED, attempts, "Claude pass rate was zero")
        return AdversarialResult(task_dir.name, target.name, TaskStatus.ACCEPTED, attempts, "mock target pass rate was zero")

    def run_many(
        self,
        tasks_root: Path,
        validations: list[ValidationResult],
        target_name: str | None = None,
    ) -> list[AdversarialResult]:
        by_id = {validation.task_id: validation for validation in validations}
        results = []
        for task_dir in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
            validation = by_id.get(task_dir.name)
            if validation is None:
                continue
            results.append(self.run_for_task(task_dir, validation, target_name))
        return results

    def _target(self, target_name: str):
        if target_name in {"claude", "claude-harness"}:
            return TemplateHarnessTarget(
                template_root=self.config.template_root,
                repo_root=repo_root(),
                timeout_sec=self.config.harness_timeout_sec,
                pass_threshold=self.config.oracle_threshold,
            )
        if target_name == "claude-api":
            return ClaudeTarget()
        if target_name.startswith("mock"):
            return mock_target(target_name)
        raise ValueError(f"Unknown target model: {target_name}")

