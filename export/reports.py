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
            lines.append(f"- Hidden variant axes: `{', '.join(candidate.hidden_variant_axes)}`")
            lines.append(f"- Hidden private variants: `{len(candidate.hidden_variants)}`")
            if candidate.zip_path:
                lines.append(f"- Zip: `{candidate.zip_path}`")
            anchor_validation = (candidate.generation_report or {}).get("anchor_validation")
            if anchor_validation is not None:
                if anchor_validation.get("passed"):
                    lines.append("- Anchor validation: `passed`")
                else:
                    failing = ", ".join(item.get("variant", "?") for item in anchor_validation.get("failing_variants", []))
                    lines.append(f"- Anchor validation: `failed` (oracle metrics_ok false for: {failing})")
                    if anchor_validation.get("reason"):
                        lines.append(f"- Anchor validation reason: {anchor_validation['reason']}")
            if validation:
                lines.append(f"- Oracle score: `{validation.oracle.score}`")
                lines.append(f"- Naive score: `{validation.naive.score}`")
                if validation.issues:
                    lines.append(f"- Validation issues: `{'; '.join(validation.issues)}`")
                combined_log = validation.oracle.details.get("combined_log") if validation.oracle.details else None
                if combined_log:
                    lines.append(f"- Oracle harness log: `{combined_log}`")
                naive_log = validation.naive.details.get("combined_log") if validation.naive.details else None
                if naive_log:
                    lines.append(f"- Naive harness log: `{naive_log}`")
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

