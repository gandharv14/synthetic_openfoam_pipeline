from __future__ import annotations

from synthetic_openfoam_pipeline.utils.templates import COMPUTE_SCORE, GRADER_LIB


class GraderWriterAgent:
    def compute_score_source(self) -> str:
        return COMPUTE_SCORE

    def grader_lib_source(self) -> str:
        return GRADER_LIB

