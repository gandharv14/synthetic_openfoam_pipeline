from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from synthetic_openfoam_pipeline.schemas import TargetAttemptResult
from synthetic_openfoam_pipeline.utils.fs import write_json


class ClaudeTarget:
    name = "claude"

    def __init__(self, timeout_sec: int = 120) -> None:
        self.timeout_sec = timeout_sec
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def attempt_task(self, task_dir: Path, attempt: int, reports_dir: Path) -> TargetAttemptResult:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        reports_dir.mkdir(parents=True, exist_ok=True)
        transcript = reports_dir / f"attempt_{attempt}.json"
        if not api_key:
            write_json(
                transcript,
                {
                    "task_id": task_dir.name,
                    "target": self.name,
                    "available": False,
                    "reason": "ANTHROPIC_API_KEY is not set",
                },
            )
            return TargetAttemptResult(
                target=self.name,
                attempt=attempt,
                passed=False,
                available=False,
                status="needs_claude_validation",
                transcript_path=str(transcript),
                error="ANTHROPIC_API_KEY is not set",
            )

        instruction = (task_dir / "instruction.md").read_text()
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "You are evaluating whether you can solve this OpenFOAM RL task. "
                        "Describe a concise repair plan only; do not claim success unless you can produce the required archive.\n\n"
                        + instruction
                    ),
                }
            ],
        }
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            write_json(transcript, {"task_id": task_dir.name, "target": self.name, "error": str(exc)})
            return TargetAttemptResult(
                target=self.name,
                attempt=attempt,
                passed=False,
                available=True,
                status="error",
                transcript_path=str(transcript),
                error=str(exc),
            )
        write_json(
            transcript,
            {
                "task_id": task_dir.name,
                "target": self.name,
                "model": self.model,
                "response": body,
                "passed": False,
                "note": "Adapter records Claude reasoning only; no generated archive was produced by this API-only attempt.",
            },
        )
        return TargetAttemptResult(target=self.name, attempt=attempt, passed=False, score=0.0, transcript_path=str(transcript))

