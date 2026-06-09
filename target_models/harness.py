from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from synthetic_openfoam_pipeline.schemas import TargetAttemptResult
from synthetic_openfoam_pipeline.utils.fs import write_json


class TemplateHarnessTarget:
    name = "claude-harness"

    def __init__(
        self,
        *,
        template_root: Path,
        repo_root: Path,
        timeout_sec: int = 1800,
        pass_threshold: float = 0.999,
    ) -> None:
        self.template_root = template_root
        self.repo_root = repo_root
        self.timeout_sec = timeout_sec
        self.pass_threshold = pass_threshold

    def attempt_task(self, task_dir: Path, attempt: int, reports_dir: Path) -> TargetAttemptResult:
        reports_dir.mkdir(parents=True, exist_ok=True)
        transcript = reports_dir / f"attempt_{attempt}.json"

        mock = os.environ.get("SYNTHETIC_OPENFOAM_MOCK_HARNESS_RESULT")
        if mock:
            return self._mock_attempt(task_dir, attempt, transcript, mock)

        problem_dir = f"problems/{task_dir.name}"
        command = (
            "set -a; "
            "if [ -f ../.env.local ]; then source ../.env.local; fi; "
            "set +a; "
            'if [ -z "${ANTHROPIC_API_KEY:-}" ]; then echo "ANTHROPIC_API_KEY missing"; exit 86; fi; '
            'if [ -z "${LBX_RL_HARNESS_MODEL:-}" ]; then export LBX_RL_HARNESS_MODEL="${ANTHROPIC_MODEL:-claude-opus-4-7}"; fi; '
            'if [ -n "${ANTHROPIC_MODEL:-}" ]; then '
            'case "${ANTHROPIC_MODEL}" in anthropic:*|openai:*|google:*) ;; *) export ANTHROPIC_MODEL="anthropic:${ANTHROPIC_MODEL}" ;; esac; '
            "fi; "
            f"uv run lbx-rl-harness run --problem-dir {problem_dir} --runtime agent"
        )
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=self.template_root,
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            write_json(
                transcript,
                {
                    "task_id": task_dir.name,
                    "target": self.name,
                    "attempt": attempt,
                    "status": "timeout",
                    "stdout": self._text(exc.stdout),
                    "stderr": self._text(exc.stderr),
                    "command": "uv run lbx-rl-harness run --problem-dir <redacted> --runtime agent",
                },
            )
            return TargetAttemptResult(
                target=self.name,
                attempt=attempt,
                passed=False,
                available=True,
                status="timeout",
                transcript_path=str(transcript),
                error="template harness timed out",
            )

        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        score = self._max_score(output)
        passed = score >= self.pass_threshold
        if proc.returncode == 86:
            status = "needs_claude_validation"
        elif proc.returncode == 0:
            status = "completed"
        else:
            status = "error"
        write_json(
            transcript,
            {
                "task_id": task_dir.name,
                "target": self.name,
                "attempt": attempt,
                "returncode": proc.returncode,
                "status": status,
                "score": score,
                "passed": passed,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "command": "uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime agent",
            },
        )
        if proc.returncode == 86:
            return TargetAttemptResult(
                target=self.name,
                attempt=attempt,
                passed=False,
                available=False,
                status=status,
                score=0.0,
                transcript_path=str(transcript),
                error="ANTHROPIC_API_KEY missing",
            )
        if proc.returncode != 0:
            return TargetAttemptResult(
                target=self.name,
                attempt=attempt,
                passed=False,
                available=True,
                status=status,
                score=score,
                transcript_path=str(transcript),
                error=f"harness exit code {proc.returncode}",
            )
        return TargetAttemptResult(
            target=self.name,
            attempt=attempt,
            passed=passed,
            available=True,
            status=status,
            score=score,
            transcript_path=str(transcript),
            error=None,
        )

    def _mock_attempt(self, task_dir: Path, attempt: int, transcript: Path, mock: str) -> TargetAttemptResult:
        if mock == "missing_key":
            write_json(transcript, {"task_id": task_dir.name, "target": self.name, "status": "needs_claude_validation"})
            return TargetAttemptResult(
                target=self.name,
                attempt=attempt,
                passed=False,
                available=False,
                status="needs_claude_validation",
                transcript_path=str(transcript),
                error="ANTHROPIC_API_KEY missing",
            )
        if mock == "timeout":
            write_json(transcript, {"task_id": task_dir.name, "target": self.name, "status": "timeout"})
            return TargetAttemptResult(
                target=self.name,
                attempt=attempt,
                passed=False,
                available=True,
                status="timeout",
                transcript_path=str(transcript),
                error="template harness timed out",
            )
        passed = mock == "pass"
        score = 1.0 if passed else 0.0
        write_json(
            transcript,
            {"task_id": task_dir.name, "target": self.name, "status": "completed", "passed": passed, "score": score},
        )
        return TargetAttemptResult(
            target=self.name,
            attempt=attempt,
            passed=passed,
            available=True,
            status="completed",
            score=score,
            transcript_path=str(transcript),
        )

    def _max_score(self, text: str) -> float:
        scores = [float(match) for match in re.findall(r'(?i)"?(?:score|reward)"?\s*[:=]\s*([01](?:\.\d+)?)', text)]
        return max(scores) if scores else 0.0

    def _text(self, value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value

