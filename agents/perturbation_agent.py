from __future__ import annotations

import importlib
import shutil
from dataclasses import asdict
from pathlib import Path

from synthetic_openfoam_pipeline.agents.skill_mapper_agent import SkillMapperAgent
from synthetic_openfoam_pipeline.openfoam.case_model import write_seed_case
from synthetic_openfoam_pipeline.schemas import ForbiddenEdit, TaskCandidate
from synthetic_openfoam_pipeline.task_families.common import apply_oracle_marker, apply_starter_perturbation
from synthetic_openfoam_pipeline.utils.hashing import sha256_file, task_id_for


class PerturbationAgent:
    def __init__(self) -> None:
        self.skills = SkillMapperAgent()

    def generate_candidate(self, *, family: str, seed: int, index: int, work_root: Path) -> TaskCandidate:
        module = importlib.import_module(f"synthetic_openfoam_pipeline.task_families.{family}")
        definition = module.build("")
        seed_spec = definition.task_spec.seed_case
        task_id = task_id_for(seed, family, seed_spec.case_kind, index)

        seed_dir = work_root / task_id / "seed"
        starter_dir = work_root / task_id / "starter"
        oracle_dir = work_root / task_id / "oracle"
        write_seed_case(seed_dir, seed_spec)
        if starter_dir.exists():
            shutil.rmtree(starter_dir)
        if oracle_dir.exists():
            shutil.rmtree(oracle_dir)
        shutil.copytree(seed_dir, starter_dir)
        shutil.copytree(seed_dir, oracle_dir)
        apply_starter_perturbation(starter_dir, definition)
        apply_oracle_marker(oracle_dir, definition)

        forbidden = [
            ForbiddenEdit(path=edit.path, sha256=sha256_file(starter_dir / edit.path), description=edit.description)
            for edit in definition.task_spec.forbidden_edits
        ]
        expected = definition.expected_metrics
        candidate = TaskCandidate(
            task_id=task_id,
            family=family,
            seed=seed,
            seed_case=seed_spec,
            skill_graph=self.skills.map_skills(family),
            hidden_variants=definition.hidden_variants,
            expected_metrics=expected,
            forbidden_edits=forbidden,
            prompt=definition.prompt,
            rationale=definition.rationale,
            starter_case_dir=starter_dir,
            oracle_case_dir=oracle_dir,
            task_spec=definition.task_spec,
            generation_report={
                "family_title": definition.title,
                "symptom": definition.symptom,
                "expected_metrics": asdict(expected),
                "claude_hard_rationale": definition.rationale,
            },
        )
        return candidate

