from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from synthetic_openfoam_pipeline.schemas import TaskCandidate


REQUIRED_INSTRUCTION_SNIPPETS = [
    "/tmp/output/fixed_case.tar.gz",
    "starter_case.tar.gz",
    "0/",
    "constant/",
    "system/",
]

PRIVATE_LEAK_PATTERNS = [
    ".synthetic_openfoam",
    "repair_state",
    "oracle",
    "scorer/data",
    "hidden_variants.json",
    "expected_metrics.json",
    "forbidden_edits.json",
]


@dataclass(frozen=True)
class PromptRewrite:
    instruction: str
    description: str
    model: str | None
    status: str
    reason: str | None = None

    def to_report(self) -> dict[str, str | None]:
        return {
            "model": self.model,
            "status": self.status,
            "reason": self.reason,
            "description": self.description,
        }


class PromptWriterAgent:
    def __init__(self, *, model: str | None = None, timeout_sec: int = 120) -> None:
        self.model = self._normalise_model(
            model
            or os.environ.get("SYNTHETIC_OPENFOAM_PROMPT_MODEL")
            or os.environ.get("ANTHROPIC_MODEL")
            or "claude-sonnet-4-5"
        )
        self.timeout_sec = timeout_sec

    def render_instruction(self, candidate: TaskCandidate) -> str:
        return self.rewrite(candidate).instruction

    def rewrite(self, candidate: TaskCandidate) -> PromptRewrite:
        original_description = self._original_description(candidate)
        api_key = self._api_key()
        if not api_key:
            return PromptRewrite(
                instruction=candidate.prompt,
                description=original_description,
                model=self.model,
                status="skipped",
                reason="ANTHROPIC_API_KEY is not configured",
            )

        try:
            body = self._call_model(api_key, candidate, original_description)
            parsed = self._parse_response(body)
            instruction = self._clean_instruction(parsed.get("instruction_md", ""))
            description = self._clean_description(parsed.get("description", ""))
            self._validate_rewrite(instruction, description)
        except Exception as exc:  # noqa: BLE001 - generation should remain auditable.
            return PromptRewrite(
                instruction=candidate.prompt,
                description=original_description,
                model=self.model,
                status="fallback",
                reason=str(exc),
            )

        return PromptRewrite(
            instruction=instruction,
            description=description,
            model=self.model,
            status="rewritten",
        )

    def _api_key(self) -> str:
        return os.environ.get("ANTHROPIC_API_KEY", "").strip()

    def _call_model(self, api_key: str, candidate: TaskCandidate, description: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": 2400,
            "temperature": 0.6,
            "system": (
                "You rewrite synthetic OpenFOAM repair tasks for an autonomous coding agent. "
                "Keep the task fair, concrete, and unambiguous. Do not reveal private grader "
                "files, oracle markers, exact hidden test values, or the scripted solution."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": self._rewrite_request(candidate, description),
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
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"prompt rewrite request failed with HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"prompt rewrite request failed: {exc}") from exc

    def _rewrite_request(self, candidate: TaskCandidate, description: str) -> str:
        hidden_axes = [variant.name.replace("_", " ") for variant in candidate.hidden_variants]
        forbidden_guidance = [
            item.description.replace("Must preserve", "Preserve").strip()
            for item in candidate.forbidden_edits
        ]
        expected = candidate.expected_metrics
        return json.dumps(
            {
                "task_id": candidate.task_id,
                "family": candidate.family,
                "current_description": description,
                "current_instruction_md": candidate.prompt,
                "skill_context": candidate.skill_graph,
                "hidden_generalization_axes": hidden_axes,
                "public_constraints": {
                    "required_artifact": "/tmp/output/fixed_case.tar.gz",
                    "starter_archive": "/data/starter_case.tar.gz",
                    "case_folders": ["0/", "constant/", "system/"],
                    "preserve_physical_inputs": forbidden_guidance,
                    "required_postprocessing": expected.postprocessing_required,
                    "source_target": expected.source_target,
                    "source_tolerance": expected.tolerance if expected.source_target is not None else None,
                },
                "response_contract": {
                    "format": "Return only valid JSON.",
                    "schema": {
                        "description": "One sentence, human-written task summary for metadata.",
                        "instruction_md": "A complete Markdown instruction for the agent.",
                    },
                },
                "rewrite_rules": [
                    "Sound like a human engineering lead filing a repair ticket, not a template.",
                    "Keep every requirement needed to solve and submit the task.",
                    "Tell the agent to inspect and repair the starter OpenFOAM case, not to edit grader files.",
                    "Mention hidden tests only qualitatively: the fix must generalize to related operating-condition changes.",
                    "Do not include exact hidden values, private scorer paths, oracle marker names, or implementation details of the generated solution.",
                    "Do not make the objective vague; name the concrete repair goal for this family.",
                ],
            },
            indent=2,
            sort_keys=True,
        )

    def _parse_response(self, body: dict[str, Any]) -> dict[str, str]:
        parts: list[str] = []
        for item in body.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        raw = "\n".join(parts).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("prompt rewrite response was not a JSON object")
        return {str(key): str(value) for key, value in data.items()}

    def _validate_rewrite(self, instruction: str, description: str) -> None:
        if len(description) < 40 or len(description) > 240:
            raise ValueError("rewritten description must be 40-240 characters")
        if len(instruction) < 500:
            raise ValueError("rewritten instruction is too short to be unambiguous")
        lowered = instruction.lower()
        missing = [snippet for snippet in REQUIRED_INSTRUCTION_SNIPPETS if snippet.lower() not in lowered]
        if missing:
            raise ValueError(f"rewritten instruction missing required snippets: {', '.join(missing)}")
        leaks = [pattern for pattern in PRIVATE_LEAK_PATTERNS if pattern.lower() in lowered]
        if leaks:
            raise ValueError(f"rewritten instruction exposes private implementation details: {', '.join(leaks)}")

    def _original_description(self, candidate: TaskCandidate) -> str:
        return candidate.generation_report.get("symptom", f"Synthetic OpenFOAM {candidate.family} task.")

    def _clean_instruction(self, instruction: str) -> str:
        return instruction.strip() + "\n"

    def _clean_description(self, description: str) -> str:
        return " ".join(description.split()).strip().replace('"', "'")

    def _normalise_model(self, model: str) -> str:
        model = model.strip()
        if model.startswith("anthropic:"):
            return model.split(":", 1)[1]
        return model

