from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from synthetic_openfoam_pipeline.agents.grader_writer_agent import GraderWriterAgent
from synthetic_openfoam_pipeline.config import PipelineConfig
from synthetic_openfoam_pipeline.schemas import TaskCandidate
from synthetic_openfoam_pipeline.utils.fs import chmod_executable, copytree_clean, ensure_dir, make_tar_gz, write_json
from synthetic_openfoam_pipeline.utils.templates import (
    dockerfile,
    naive_script,
    render_script,
    solution_script,
    task_toml,
    test_script,
)


class LbxTaskExporter:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.grader = GraderWriterAgent()

    def export(self, candidate: TaskCandidate) -> Path:
        task_dir = self.config.problems_root / candidate.task_id
        template_write_root = self.config.template_write_root
        ensure_dir(task_dir, allow_template_output=template_write_root)
        for sub in [
            "environment",
            "data",
            "scorer/data",
            "solution",
            "baselines",
            "tests",
        ]:
            ensure_dir(task_dir / sub, allow_template_output=template_write_root)

        description = candidate.generation_report.get(
            "public_description",
            candidate.generation_report.get("symptom", f"Synthetic OpenFOAM {candidate.family} task."),
        )
        (task_dir / "task.toml").write_text(task_toml(candidate.task_id, description.replace('"', "'")))
        write_json(
            task_dir / "metadata.json",
            {
                "benchmark": "taiga_task",
                "problem_data": {"instance_id": candidate.task_id, "description": description},
            },
            allow_template_output=template_write_root,
        )
        (task_dir / "instruction.md").write_text(candidate.prompt)
        (task_dir / "README.md").write_text(self._readme(candidate))
        problem_dir = self._docker_problem_dir(candidate.task_id)
        (task_dir / "environment" / "Dockerfile").write_text(dockerfile(problem_dir))

        make_tar_gz(
            candidate.starter_case_dir,
            task_dir / "data" / "starter_case.tar.gz",
            arcname="starter_case",
            allow_template_output=template_write_root,
        )
        make_tar_gz(
            candidate.starter_case_dir,
            task_dir / "scorer" / "data" / "pristine_case.tar.gz",
            arcname="pristine_case",
            allow_template_output=template_write_root,
        )
        write_json(
            task_dir / "data" / "task_manifest.json",
            {
                "task_id": candidate.task_id,
                "public_note": "Repair the OpenFOAM case described in the instruction. Private validation variants and expected metrics are held by the scorer.",
            },
            allow_template_output=template_write_root,
        )
        write_json(
            task_dir / "scorer" / "data" / "hidden_variants.json",
            [asdict(v) for v in candidate.hidden_variants],
            allow_template_output=template_write_root,
        )
        expected = asdict(candidate.expected_metrics)
        expected["family"] = candidate.family
        write_json(
            task_dir / "scorer" / "data" / "expected_metrics.json",
            expected,
            allow_template_output=template_write_root,
        )
        write_json(
            task_dir / "scorer" / "data" / "forbidden_edits.json",
            [asdict(e) for e in candidate.forbidden_edits],
            allow_template_output=template_write_root,
        )
        (task_dir / "scorer" / "__init__.py").write_text("")
        (task_dir / "scorer" / "compute_score.py").write_text(self.grader.compute_score_source())
        (task_dir / "scorer" / "openfoam_grader_lib.py").write_text(self.grader.grader_lib_source())

        marker = {"repair_state": "oracle", "family": candidate.family, "task_id": candidate.task_id}
        (task_dir / "solution" / "solve.sh").write_text(solution_script(candidate.task_id, candidate.family, marker))
        (task_dir / "solution" / "solution_notes.md").write_text(self._solution_notes(candidate))
        (task_dir / "solution" / "render.sh").write_text(render_script())
        (task_dir / "baselines" / "naive.sh").write_text(naive_script())
        (task_dir / "tests" / "test.sh").write_text(test_script())
        for path in [
            task_dir / "solution" / "solve.sh",
            task_dir / "solution" / "render.sh",
            task_dir / "baselines" / "naive.sh",
            task_dir / "tests" / "test.sh",
        ]:
            chmod_executable(path)

        candidate.task_dir = task_dir
        report = candidate.to_manifest()
        report["output_contract"] = "/tmp/output/fixed_case.tar.gz"
        write_json(task_dir / "generation_report.json", report, allow_template_output=template_write_root)
        self._mirror_task_dir(task_dir)
        return task_dir

    def _mirror_task_dir(self, task_dir: Path) -> None:
        mirror_root = self.config.mirror_problems_root
        if mirror_root is None:
            return
        mirror_task_dir = mirror_root / task_dir.name
        if mirror_task_dir.resolve() == task_dir.resolve():
            return
        copytree_clean(task_dir, mirror_task_dir, allow_template_output=self.config.template_write_root)

    def _docker_problem_dir(self, task_id: str) -> str:
        if self.config.use_template_problems and self.config.problems_root == self.config.template_problems_root:
            return f"problems/{task_id}"
        return f"{self.config.problems_root.as_posix()}/{task_id}"

    def _readme(self, candidate: TaskCandidate) -> str:
        return f"""# {candidate.task_id}

Synthetic OpenFOAM RL task generated by the root-level pipeline.

Family: `{candidate.family}`

The agent must repair the public starter case from `data/starter_case.tar.gz` and
write `/tmp/output/fixed_case.tar.gz`. The deterministic scorer applies hidden
variants and checks OpenFOAM-style structure, forbidden edits, solver health, and
engineering metrics.

Claude-hard rationale: {candidate.rationale}
"""

    def _solution_notes(self, candidate: TaskCandidate) -> str:
        return f"""# Oracle Notes

The oracle starts from the public starter case, applies the minimal repair for
`{candidate.family}`, and packages the corrected case as `/tmp/output/fixed_case.tar.gz`.

It is expected to score 1.0 when validated through the template Docker harness
with real OpenFOAM available in the task image.
"""

