from __future__ import annotations

import shutil
from dataclasses import dataclass

from synthetic_openfoam_pipeline.agents.adversarial_filter_agent import AdversarialFilterAgent
from synthetic_openfoam_pipeline.agents.anchor_precompute_agent import AnchorPrecomputeAgent
from synthetic_openfoam_pipeline.agents.curator_agent import CuratorAgent
from synthetic_openfoam_pipeline.agents.oracle_builder_agent import OracleBuilderAgent
from synthetic_openfoam_pipeline.agents.perturbation_agent import PerturbationAgent
from synthetic_openfoam_pipeline.agents.prompt_writer_agent import PromptWriterAgent
from synthetic_openfoam_pipeline.agents.robustness_verifier_agent import RobustnessVerifierAgent
from synthetic_openfoam_pipeline.config import PipelineConfig
from synthetic_openfoam_pipeline.export.lbx_task_exporter import LbxTaskExporter
from synthetic_openfoam_pipeline.export.reports import ReportWriter
from synthetic_openfoam_pipeline.export.zip_exporter import ZipExporter
from synthetic_openfoam_pipeline.schemas import AdversarialResult, TaskCandidate, ValidationResult
from synthetic_openfoam_pipeline.utils.fs import ensure_dir


@dataclass
class DeploymentResult:
    candidates: list[TaskCandidate]
    validations: list[ValidationResult]
    adversarial: list[AdversarialResult]

    @property
    def task_ids(self) -> list[str]:
        return [candidate.task_id for candidate in self.candidates]


def generate_candidates(config: PipelineConfig) -> list[TaskCandidate]:
    ensure_dir(config.output_root, allow_template_output=config.allow_template_output)
    work_root = config.reports_root / "_work"
    if work_root.exists():
        shutil.rmtree(work_root)
    ensure_dir(work_root, allow_template_output=config.allow_template_output)
    perturb = PerturbationAgent(difficulty=config.difficulty)
    oracle = OracleBuilderAgent()
    anchors = AnchorPrecomputeAgent()
    prompt_writer = PromptWriterAgent(model=config.prompt_rewrite_model) if config.prompt_rewrite_enabled else None
    candidates: list[TaskCandidate] = []
    for i in range(config.count):
        family = config.families[i % len(config.families)]
        candidate = oracle.attach_oracle(perturb.generate_candidate(family=family, seed=config.seed, index=i, work_root=work_root))
        anchors.attach_anchors(candidate)
        if prompt_writer is not None:
            rewrite = prompt_writer.rewrite(candidate)
            candidate.generation_report["original_prompt"] = candidate.prompt
            candidate.generation_report["prompt_rewrite"] = rewrite.to_report()
            candidate.generation_report["public_description"] = rewrite.description
            candidate.prompt = rewrite.instruction
        else:
            candidate.generation_report["prompt_rewrite"] = {
                "status": "disabled",
                "model": None,
                "reason": "prompt rewriting disabled by config",
            }
        candidates.append(candidate)
    return CuratorAgent().dedupe(candidates)


def export_candidates(config: PipelineConfig, candidates: list[TaskCandidate]) -> None:
    exporter = LbxTaskExporter(config)
    zipper = ZipExporter(config)
    for candidate in candidates:
        exporter.export(candidate)
        if config.zip_tasks:
            zipper.export(candidate)


def deploy_handful(config: PipelineConfig) -> DeploymentResult:
    candidates = generate_candidates(config)
    export_candidates(config, candidates)

    verifier = RobustnessVerifierAgent(config)
    validations = verifier.validate_many(config.problems_root)
    curator = CuratorAgent()
    curator.apply_validation(candidates, validations)

    adversarial = AdversarialFilterAgent(config).run_many(config.problems_root, validations, config.target)
    curator.apply_adversarial(candidates, adversarial)
    ReportWriter(config).write(candidates, validations, adversarial)
    return DeploymentResult(candidates, validations, adversarial)

