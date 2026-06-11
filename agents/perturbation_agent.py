from __future__ import annotations

import importlib
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from synthetic_openfoam_pipeline.agents.skill_mapper_agent import SkillMapperAgent
from synthetic_openfoam_pipeline.openfoam.case_model import write_seed_case
from synthetic_openfoam_pipeline.openfoam.evidence import collect_symptom_evidence
from synthetic_openfoam_pipeline.schemas import CorruptionSpec, ForbiddenEdit, OracleRepairSpec, TaskCandidate
from synthetic_openfoam_pipeline.task_families.common import (
    DIFFICULTY_CORRUPTION_COUNTS,
    SYMPTOM_EVIDENCE_PLACEHOLDER,
    apply_corruption_steps,
    apply_oracle_marker,
    apply_starter_perturbation,
    family_definition,
    select_hidden_variants,
    select_secondary_corruption_families,
    variant_axes,
)
from synthetic_openfoam_pipeline.utils.hashing import sha256_file, task_id_for


class PerturbationAgent:
    def __init__(self, difficulty: str = "easy") -> None:
        if difficulty not in DIFFICULTY_CORRUPTION_COUNTS:
            raise ValueError(f"difficulty must be one of {sorted(DIFFICULTY_CORRUPTION_COUNTS)}, got {difficulty!r}")
        self.difficulty = difficulty
        self.skills = SkillMapperAgent()

    def generate_candidate(self, *, family: str, seed: int, index: int, work_root: Path) -> TaskCandidate:
        module = importlib.import_module(f"synthetic_openfoam_pipeline.task_families.{family}")
        seed_selector = {"seed": seed, "index": index}
        definition = module.build("", seed_selector=seed_selector)
        seed_spec = definition.task_spec.seed_case
        task_id = task_id_for(seed, family, seed_spec.case_kind, index)

        secondary_names = select_secondary_corruption_families(
            family,
            seed_spec.case_kind,
            seed=seed,
            index=index,
            count=DIFFICULTY_CORRUPTION_COUNTS[self.difficulty] - 1,
        )
        # Secondary corruptions are pinned to registry defaults so their edits
        # match the primary family's seed dictionaries.
        secondary_definitions = [
            family_definition(name, "", seed_selector, goal_targets=False) for name in secondary_names
        ]

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
        for secondary in secondary_definitions:
            apply_corruption_steps(starter_dir, secondary.task_spec.corruption.steps)
        apply_oracle_marker(oracle_dir, definition)

        # Combined spec: corruption steps in application order; the oracle
        # repair is the exact reverse of every applied step, unwound in
        # reverse order so the stack stays invertible.
        corruption_steps = list(definition.task_spec.corruption.steps)
        repair_steps = list(definition.task_spec.oracle_repair.steps)
        for secondary in secondary_definitions:
            corruption_steps.extend(secondary.task_spec.corruption.steps)
            repair_steps = list(secondary.task_spec.oracle_repair.steps) + repair_steps
        task_spec = replace(
            definition.task_spec,
            corruption=CorruptionSpec(corruption_steps, definition.task_spec.corruption.description),
            oracle_repair=OracleRepairSpec(repair_steps, definition.task_spec.oracle_repair.description),
        )

        # Run the broken starter once and embed the diagnostic excerpt as the
        # public symptom evidence; the agent diagnoses from logs, not titles.
        evidence = collect_symptom_evidence(starter_dir)
        prompt = definition.prompt.replace(SYMPTOM_EVIDENCE_PLACEHOLDER, evidence["excerpt"])

        forbidden = [
            ForbiddenEdit(path=edit.path, sha256=sha256_file(starter_dir / edit.path), description=edit.description)
            for edit in definition.task_spec.forbidden_edits
        ]
        expected = definition.expected_metrics
        hidden_variants = select_hidden_variants(definition.task_spec.hidden_variant_grid, family=family, seed=seed, index=index)
        hidden_variant_axes = variant_axes(definition.task_spec.hidden_variant_grid)
        candidate = TaskCandidate(
            task_id=task_id,
            family=family,
            seed=seed,
            seed_case=seed_spec,
            skill_graph=self.skills.map_skills(family),
            hidden_variants=hidden_variants,
            expected_metrics=expected,
            forbidden_edits=forbidden,
            prompt=prompt,
            rationale=definition.rationale,
            starter_case_dir=starter_dir,
            oracle_case_dir=oracle_dir,
            task_spec=task_spec,
            corruption_families=[family, *secondary_names],
            generation_report={
                "family_title": definition.title,
                "symptom": definition.symptom,
                "symptom_evidence": evidence,
                "difficulty": self.difficulty,
                "corruption_families": [family, *secondary_names],
                "expected_metrics": asdict(expected),
                "hidden_variant_axes": hidden_variant_axes,
                "hidden_variant_count": len(hidden_variants),
                "hidden_variant_grid_size": len(definition.task_spec.hidden_variant_grid),
                "claude_hard_rationale": definition.rationale,
            },
        )
        return candidate

