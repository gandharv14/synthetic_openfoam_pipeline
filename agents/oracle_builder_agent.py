from __future__ import annotations

from synthetic_openfoam_pipeline.schemas import TaskCandidate


class OracleBuilderAgent:
    def attach_oracle(self, candidate: TaskCandidate) -> TaskCandidate:
        candidate.generation_report["oracle_strategy"] = (
            "Extract starter_case.tar.gz, apply the family-specific OpenFOAM dictionary repair, "
            "then package the corrected case as /tmp/output/fixed_case.tar.gz."
        )
        return candidate

