from __future__ import annotations

from pathlib import Path

from synthetic_openfoam_pipeline.config import PipelineConfig
from synthetic_openfoam_pipeline.schemas import AdversarialResult, TaskCandidate, ValidationResult
from synthetic_openfoam_pipeline.utils.fs import ensure_dir, write_json


class ReportWriter:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def write(
        self,
        candidates: list[TaskCandidate],
        validations: list[ValidationResult] | None = None,
        adversarial: list[AdversarialResult] | None = None,
    ) -> None:
        ensure_dir(self.config.output_root, allow_template_output=self.config.allow_template_output)
        validations = validations or []
        adversarial = adversarial or []
        manifest = {
            "output_root": str(self.config.output_root),
            "tasks": [candidate.to_manifest() for candidate in candidates],
            "validations": [result.to_dict() for result in validations],
            "adversarial": [result.to_dict() for result in adversarial],
        }
        write_json(self.config.output_root / "manifest.json", manifest, allow_template_output=self.config.allow_template_output)
        (self.config.output_root / "report.md").write_text(self._markdown(candidates, validations, adversarial))

    def _markdown(
        self,
        candidates: list[TaskCandidate],
        validations: list[ValidationResult],
        adversarial: list[AdversarialResult],
    ) -> str:
        lines = ["# Synthetic OpenFOAM Task Generation Report", ""]
        lines.append(f"Generated tasks: {len(candidates)}")
        lines.append("")
        for candidate in candidates:
            validation = next((v for v in validations if v.task_id == candidate.task_id), None)
            adv = next((a for a in adversarial if a.task_id == candidate.task_id), None)
            lines.append(f"## {candidate.task_id}")
            lines.append(f"- Family: `{candidate.family}`")
            lines.append(f"- Status: `{candidate.status.value}`")
            lines.append(f"- Task dir: `{candidate.task_dir}`")
            if candidate.zip_path:
                lines.append(f"- Zip: `{candidate.zip_path}`")
            if validation:
                lines.append(f"- Oracle score: `{validation.oracle.score}`")
                lines.append(f"- Naive score: `{validation.naive.score}`")
            if adv:
                lines.append(f"- Adversarial target: `{adv.target}`")
                lines.append(f"- Adversarial status: `{adv.status.value}`")
                lines.append(f"- Reason: {adv.reason}")
            lines.append(f"- Claude-hard rationale: {candidate.rationale}")
            lines.append("")
        return "\n".join(lines)


def read_manifest(path: Path) -> dict:
    import json

    return json.loads(path.read_text())

