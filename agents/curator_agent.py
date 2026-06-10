from __future__ import annotations

from synthetic_openfoam_pipeline.schemas import AdversarialResult, TaskCandidate, TaskStatus, ValidationResult


class CuratorAgent:
    def apply_validation(self, candidates: list[TaskCandidate], validations: list[ValidationResult]) -> None:
        by_id = {v.task_id: v for v in validations}
        for candidate in candidates:
            result = by_id.get(candidate.task_id)
            if result:
                candidate.generation_report["validation"] = result.to_dict()
                # A rejection recorded earlier (e.g. anchor validation at
                # export) is final; later stages must not resurrect the task.
                if candidate.status is not TaskStatus.REJECTED:
                    candidate.status = result.status

    def apply_adversarial(self, candidates: list[TaskCandidate], results: list[AdversarialResult]) -> None:
        by_id = {r.task_id: r for r in results}
        for candidate in candidates:
            result = by_id.get(candidate.task_id)
            if result:
                candidate.generation_report["adversarial"] = result.to_dict()
                if candidate.status is not TaskStatus.REJECTED:
                    candidate.status = result.status

    def dedupe(self, candidates: list[TaskCandidate]) -> list[TaskCandidate]:
        seen: set[tuple[str, str]] = set()
        kept: list[TaskCandidate] = []
        for candidate in candidates:
            key = (candidate.family, candidate.task_id)
            if key in seen:
                candidate.status = TaskStatus.REJECTED
                continue
            seen.add(key)
            kept.append(candidate)
        return kept

