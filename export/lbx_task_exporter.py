from __future__ import annotations

import json
import os
import subprocess
import textwrap
from dataclasses import asdict
from pathlib import Path

from synthetic_openfoam_pipeline.agents.grader_writer_agent import GraderWriterAgent
from synthetic_openfoam_pipeline.config import PipelineConfig
from synthetic_openfoam_pipeline.schemas import TaskCandidate, TaskStatus
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
                "hidden_variant_axes": candidate.hidden_variant_axes,
                "hidden_variant_count": len(candidate.hidden_variants),
            },
            allow_template_output=template_write_root,
        )
        write_json(
            task_dir / "scorer" / "data" / "hidden_variants.json",
            [asdict(v) for v in candidate.hidden_variants],
            allow_template_output=template_write_root,
        )
        write_json(
            task_dir / "scorer" / "data" / "metric_anchors.json",
            candidate.metric_anchors or {"version": 1, "by_variant": {}},
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

        if self._requires_real_anchors():
            anchors = self._precompute_real_anchors(task_dir, candidate)
            candidate.metric_anchors = anchors
            candidate.generation_report["metric_anchor_summary"] = {
                "version": anchors["version"],
                "mode": anchors["mode"],
                "variant_count": len(anchors["by_variant"]),
            }
            write_json(
                task_dir / "scorer" / "data" / "metric_anchors.json",
                candidate.metric_anchors,
                allow_template_output=template_write_root,
            )
            self._validate_oracle_anchor_status(candidate, anchors)

        candidate.task_dir = task_dir
        report = candidate.to_manifest()
        report["output_contract"] = "/tmp/output/fixed_case.tar.gz"
        write_json(task_dir / "generation_report.json", report, allow_template_output=template_write_root)
        self._mirror_task_dir(task_dir)
        return task_dir

    def _requires_real_anchors(self) -> bool:
        env = os.environ.get("SYNTHETIC_OPENFOAM_REQUIRE_REAL_ANCHORS")
        if env is not None:
            return env.strip().lower() in {"1", "true", "yes"}
        return self.config.real_anchors

    def _validate_oracle_anchor_status(self, candidate: TaskCandidate, anchors: dict) -> None:
        """Reject candidates whose oracle fails its own precomputed anchors.

        Real anchor precompute records oracle_status per variant; if the oracle
        run itself did not satisfy the engineering-metric tolerance bands
        (metrics_ok false), the task can never score 1.0 and must not ship.
        """
        by_variant = anchors.get("by_variant") or {}
        failing = []
        for name in sorted(by_variant):
            status = (by_variant[name] or {}).get("oracle_status") or {}
            if status.get("metrics_ok") is not True:
                failing.append({"variant": name, "oracle_status": status})
        candidate.generation_report["anchor_validation"] = {
            "passed": not failing,
            "mode": anchors.get("mode"),
            "checked_variants": len(by_variant),
            "failing_variants": failing,
        }
        if failing:
            candidate.status = TaskStatus.REJECTED
            candidate.generation_report["anchor_validation"]["reason"] = (
                "oracle solution failed its own precomputed metric anchors "
                "(oracle_status.metrics_ok is not true); task rejected at export"
            )

    def _precompute_real_anchors(self, task_dir: Path, candidate: TaskCandidate) -> dict:
        if not self.config.template_root.resolve() in task_dir.resolve().parents:
            raise RuntimeError("Real anchor precompute requires task_dir under template_root for Docker build context")
        problem_dir = self._docker_problem_dir(candidate.task_id)
        tag = f"synthetic-openfoam-anchor-{candidate.task_id}:latest"
        runtime_image = self._anchor_runtime_image()
        if runtime_image is None:
            build = subprocess.run(
                [
                    "docker",
                    "build",
                    "--pull=false",
                    "-f",
                    str((task_dir / "environment" / "Dockerfile").resolve()),
                    "--build-arg",
                    f"PROBLEM_DIR={problem_dir}",
                    "-t",
                    tag,
                    ".",
                ],
                cwd=self.config.template_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=max(int(self.config.harness_timeout_sec), 1800),
            )
            if build.returncode != 0:
                raise RuntimeError(f"Docker build failed while precomputing real anchors for {candidate.task_id}:\n{build.stdout[-4000:]}\n{build.stderr[-4000:]}")
            runtime_image = tag
        script = textwrap.dedent(
            r"""
            import copy
            import importlib.util
            import json
            import os
            import shutil
            import subprocess
            import sys
            import tarfile
            import tempfile
            from pathlib import Path

            problem = Path("/problem")
            spec = importlib.util.spec_from_file_location("openfoam_grader_lib", problem / "scorer" / "openfoam_grader_lib.py")
            grader = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = grader
            spec.loader.exec_module(grader)

            expected_base = json.loads((problem / "scorer" / "data" / "expected_metrics.json").read_text())
            variants = json.loads((problem / "scorer" / "data" / "hidden_variants.json").read_text())

            def extract_archive(archive: Path, dest: Path) -> Path:
                dest.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(dest)
                children = [p for p in dest.iterdir() if p.is_dir()]
                return children[0] if children else dest

            def run_solution(work: Path) -> Path:
                out = work / "oracle_out"
                out.mkdir()
                proc = subprocess.run(
                    ["bash", str(problem / "solution" / "solve.sh")],
                    env={**os.environ, "LBT_DATA_DIR": str(problem / "data"), "LBT_OUTPUT_DIR": str(out)},
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=180,
                )
                if proc.returncode != 0:
                    raise SystemExit(json.dumps({"error": "solution failed", "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}))
                return extract_archive(out / "fixed_case.tar.gz", work / "oracle_case")

            def control_end_time(case: Path) -> float:
                text = (case / "system" / "controlDict").read_text()
                import re
                match = re.search(r"\bendTime\s+([-+0-9.eE]+);", text)
                return float(match.group(1)) if match else 100.0

            def run_metrics(base_case: Path, variant: dict, expected: dict) -> dict:
                case = Path(tempfile.mkdtemp(prefix="anchor-case-"))
                shutil.copytree(base_case, case / "case")
                case = case / "case"
                grader.apply_hidden_variant(case, variant)
                run = grader.RealOpenFoamRunner().run(case, expected, variant)
                payload = {
                    "ok": run.ok,
                    "mesh_ok": run.mesh_ok,
                    "solver_ok": run.solver_ok,
                    "post_ok": run.post_ok,
                    "metrics_ok": run.metrics_ok,
                    "metrics": run.metrics,
                    "reason": run.reason,
                    "postprocessing": run.postprocessing or {},
                }
                shutil.rmtree(case.parent, ignore_errors=True)
                return payload

            with tempfile.TemporaryDirectory(prefix="real-anchor-") as tmp:
                tmp = Path(tmp)
                oracle_base = run_solution(tmp)
                baseline_base = extract_archive(problem / "data" / "starter_case.tar.gz", tmp / "baseline_case")
                anchors = {"version": 1, "family": expected_base.get("family"), "mode": "openfoam-precomputed", "by_variant": {}}
                for variant in variants:
                    expected = copy.deepcopy(expected_base)
                    expected.update(variant.get("expected_overrides") or {})
                    oracle_payload = run_metrics(oracle_base, variant, expected)
                    baseline_expected = copy.deepcopy(expected)
                    baseline_expected["end_time"] = control_end_time(baseline_base)
                    baseline_payload = run_metrics(baseline_base, variant, baseline_expected)
                    if not oracle_payload["solver_ok"]:
                        raise SystemExit(json.dumps({"error": "oracle anchor solver failed", "variant": variant, "payload": oracle_payload}))
                    anchors["by_variant"][variant["name"]] = {
                        "changes": variant.get("changes", {}),
                        "oracle": oracle_payload["metrics"],
                        "baseline": baseline_payload.get("metrics") or {},
                        "tolerances": {
                            "flow_rate": 0.10,
                            "pressure_drop": 0.50,
                            "source_strength": float(expected.get("tolerance") or 0.05),
                            "source_response": 0.75,
                            "mesh_cells": 0.0,
                            "mesh_grading": 0.05,
                            "force_Cd": 1.0,
                            "force_Cl": 1.0,
                            "scalar_average": 0.25,
                            "write_interval": 0.0
                        },
                        "oracle_status": {k: oracle_payload[k] for k in ("ok", "mesh_ok", "solver_ok", "post_ok", "metrics_ok", "reason")},
                        "baseline_status": {k: baseline_payload[k] for k in ("ok", "mesh_ok", "solver_ok", "post_ok", "metrics_ok", "reason")},
                    }
                print(json.dumps(anchors))
            """
        )
        run = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{task_dir.resolve()}:/problem:ro",
                runtime_image,
                "python3",
                "-c",
                script,
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=max(int(self.config.harness_timeout_sec), 1800),
        )
        if run.returncode != 0:
            raise RuntimeError(f"Real anchor precompute failed for {candidate.task_id}:\n{run.stdout[-4000:]}\n{run.stderr[-4000:]}")
        try:
            anchors = json.loads(run.stdout.strip().splitlines()[-1])
        except Exception as exc:
            raise RuntimeError(f"Could not parse real anchors for {candidate.task_id}: {exc}\n{run.stdout[-4000:]}\n{run.stderr[-4000:]}") from exc
        if anchors.get("mode") != "openfoam-precomputed" or set(anchors.get("by_variant", {})) != {variant.name for variant in candidate.hidden_variants}:
            raise RuntimeError(f"Invalid real anchor payload for {candidate.task_id}")
        return anchors

    def _anchor_runtime_image(self) -> str | None:
        configured = os.environ.get("SYNTHETIC_OPENFOAM_ANCHOR_IMAGE")
        if configured:
            return configured
        images = subprocess.run(
            ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if images.returncode != 0:
            return None
        for line in images.stdout.splitlines():
            if line.startswith("lbx-rl-harness-openfoam-") and not line.endswith(":<none>"):
                return line
        return None

    def _mirror_task_dir(self, task_dir: Path) -> None:
        mirror_root = self.config.mirror_problems_root
        if mirror_root is None:
            return
        mirror_task_dir = mirror_root / task_dir.name
        if mirror_task_dir.resolve() == task_dir.resolve():
            return
        copytree_clean(task_dir, mirror_task_dir, allow_template_output=self.config.template_write_root)

    def _docker_problem_dir(self, task_id: str) -> str:
        try:
            relative = (self.config.problems_root / task_id).resolve().relative_to(self.config.template_root.resolve())
            return relative.as_posix()
        except ValueError:
            pass
        return f"{self.config.problems_root.as_posix()}/{task_id}"

    def _readme(self, candidate: TaskCandidate) -> str:
        return f"""# {candidate.task_id}

Synthetic OpenFOAM RL task generated by the root-level pipeline.

Family: `{candidate.family}`

The agent must repair the public starter case from `data/starter_case.tar.gz` and
write `/tmp/output/fixed_case.tar.gz`. The deterministic scorer applies hidden
variants and checks OpenFOAM-style structure, forbidden edits, solver health, and
engineering metrics.

Hidden validation axes: {", ".join(candidate.hidden_variant_axes)}

Claude-hard rationale: {candidate.rationale}
"""

    def _solution_notes(self, candidate: TaskCandidate) -> str:
        return f"""# Oracle Notes

The oracle starts from the public starter case, applies the minimal repair for
`{candidate.family}`, and packages the corrected case as `/tmp/output/fixed_case.tar.gz`.

It is expected to score 1.0 when validated through the template Docker harness
with real OpenFOAM available in the task image.
"""

