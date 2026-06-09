from __future__ import annotations

from pathlib import Path

from synthetic_openfoam_pipeline.openfoam.case_model import pick_seed_for_family, write_seed_case
from synthetic_openfoam_pipeline.schemas import SeedCaseSpec


class CorpusAgent:
    def select_seed(self, family: str) -> SeedCaseSpec:
        return pick_seed_for_family(family)

    def write_seed(self, family: str, destination: Path) -> SeedCaseSpec:
        spec = self.select_seed(family)
        write_seed_case(destination, spec)
        return spec

