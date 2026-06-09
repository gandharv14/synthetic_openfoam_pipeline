from __future__ import annotations

import json
from textwrap import dedent


def task_toml(task_id: str, description: str) -> str:
    return dedent(
        f"""
        schema_version = "1.1"

        [task]
        name = "labelbox/{task_id}"
        description = "{description}"

        [environment]
        cpus = 4
        memory_mb = 8192
        storage_mb = 10000
        gpus = 0
        gpu_types = []
        allow_internet = true

        [agent]
        timeout_sec = 1800

        [verifier]
        timeout_sec = 1200
        env = []

        [difficulty]
        task_type = "cfd"
        domain = "openfoam-repair"

        [ground_truth]
        in_container = true
        render_command = "bash solution/render.sh"
        render_outputs = [
          {{ path = "/tmp/output/rendering.mp4", required = false, description = "Optional synthetic OpenFOAM reviewer artifact" }},
        ]

        [[outputs]]
        path = "/tmp/output/fixed_case.tar.gz"
        required = true
        description = "Corrected OpenFOAM case archive"
        """
    ).lstrip()


def dockerfile(problem_dir: str) -> str:
    return dedent(
        f"""
        ARG BASE_IMAGE=lbx-tasks-base
        ARG BASE_TAG=runtime-ml-core-py313-local
        ARG PROBLEM_DIR={problem_dir}
        FROM ${{BASE_IMAGE}}:${{BASE_TAG}}

        ARG PROBLEM_DIR={problem_dir}
        WORKDIR /workdir

        RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output
        COPY grader/ /mcp_server/grading/
        RUN env -u UV_SYSTEM_PYTHON uv pip install --python /mcp_server/.venv/bin/python --no-cache -e /mcp_server/grading
        RUN if [ ! -f /etc/solver-envs.d/openfoam.sh ]; then \\
              apt-get update \\
              && apt-get install -y --no-install-recommends openfoam \\
              && rm -rf /var/lib/apt/lists/* \\
              && mkdir -p /etc/solver-envs.d \\
              && printf '%s\\n' \\
                'if [ -f /usr/share/openfoam/etc/bashrc ]; then' \\
                '  source /usr/share/openfoam/etc/bashrc' \\
                'elif [ -f /opt/openfoam*/etc/bashrc ]; then' \\
                '  source /opt/openfoam*/etc/bashrc' \\
                'fi' \\
                > /etc/solver-envs.d/openfoam.sh; \\
            fi
        COPY ${{PROBLEM_DIR}}/data/ /data/
        COPY --chmod=0700 ${{PROBLEM_DIR}}/scorer/data/ /mcp_server/data/
        COPY --chmod=0700 ${{PROBLEM_DIR}}/scorer/ /mcp_server/grader/
        RUN chmod 0755 /mcp_server && chmod -R 0700 /mcp_server/data /mcp_server/grader

        COPY ${{PROBLEM_DIR}}/task.toml ${{PROBLEM_DIR}}/instruction.md /task/
        """
    ).lstrip()


def solution_script(task_id: str, family: str, marker: dict) -> str:
    _ = (task_id, marker)
    return dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        OUT="${{LBT_OUTPUT_DIR:-/tmp/output}}"
        TASK_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
        DATA_DIR="${{LBT_DATA_DIR:-/data}}"
        if [[ ! -f "${{DATA_DIR}}/starter_case.tar.gz" ]]; then
          DATA_DIR="${{TASK_DIR}}/data"
        fi
        WORK="$(mktemp -d)"
        trap 'rm -rf "${{WORK}}"' EXIT
        mkdir -p "${{OUT}}" "${{WORK}}/case"
        tar -xzf "${{DATA_DIR}}/starter_case.tar.gz" -C "${{WORK}}/case" --strip-components=1
        python3 - "${{WORK}}/case" <<'PY'
        import sys
        from pathlib import Path
        case = Path(sys.argv[1])

        FUNCTIONS = '''functions
        {{
            residuals
            {{
                type residuals;
                libs ("libutilityFunctionObjects.so");
                writeControl timeStep;
                writeInterval 1;
                fields (p U);
            }}
            probes
            {{
                type probes;
                libs ("libsampling.so");
                writeControl timeStep;
                writeInterval 1;
                fields (p U);
                probeLocations ((1 0.5 0.05));
            }}
            fieldMinMax
            {{
                type fieldMinMax;
                libs ("libfieldFunctionObjects.so");
                writeControl timeStep;
                writeInterval 1;
                fields (p U);
            }}
        }}
        '''

        def replace(path: Path, old: str, new: str) -> None:
            text = path.read_text()
            if old not in text:
                raise SystemExit(f"expected text not found in {{path}}: {{old!r}}")
            path.write_text(text.replace(old, new))

        family = {family!r}
        if family == "boundary_condition_repair":
            replace(case / "0" / "p", "outlet {{ type zeroGradient; }}", "outlet {{ type fixedValue; value uniform 0; }}")
        elif family == "postprocessing_instrumentation":
            replace(case / "system" / "controlDict", "functions {{}}", FUNCTIONS)
        elif family == "solver_convergence_repair":
            replace(case / "system" / "fvSchemes", "div(phi,U) Gauss linear;", "div(phi,U) Gauss linearUpwind grad(U);")
        elif family == "mesh_quality_repair":
            replace(case / "system" / "blockMeshDict", "simpleGrading (20 1 1)", "simpleGrading (1.5 1 1)")
        elif family == "source_term_calibration":
            replace(case / "constant" / "fvOptions", "U ((0.000 0 0) 0);", "U ((0.200 0 0) 0);")
        else:
            raise SystemExit(f"unknown family: {{family}}")
        PY
        tar -czf "${{OUT}}/fixed_case.tar.gz" -C "${{WORK}}" case
        """
    )


def naive_script() -> str:
    return dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        OUT="${LBT_OUTPUT_DIR:-/tmp/output}"
        TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        DATA_DIR="${LBT_DATA_DIR:-/data}"
        if [[ ! -f "${DATA_DIR}/starter_case.tar.gz" ]]; then
          DATA_DIR="${TASK_DIR}/data"
        fi
        WORK="$(mktemp -d)"
        trap 'rm -rf "${WORK}"' EXIT
        mkdir -p "${OUT}" "${WORK}/case"
        tar -xzf "${DATA_DIR}/starter_case.tar.gz" -C "${WORK}/case" --strip-components=1
        tar -czf "${OUT}/fixed_case.tar.gz" -C "${WORK}" case
        """
    )


def render_script() -> str:
    return dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        OUT="${LBT_OUTPUT_DIR:-/tmp/output}"
        mkdir -p "${OUT}"
        ffmpeg -y -f lavfi -i color=c=navy:s=1280x720:d=1 -pix_fmt yuv420p "${OUT}/rendering.mp4" >/dev/null 2>&1
        """
    )


def test_script() -> str:
    return dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        PRIVATE_DIR="${LBT_PRIVATE_DIR:-/mcp_server/data}"
        if [[ ! -d "${PRIVATE_DIR}" ]]; then
          PRIVATE_DIR="${TASK_DIR}/scorer/data"
        fi
        mkdir -p /tmp/openfoam-test-logs
        python3 - "${TASK_DIR}" "${PRIVATE_DIR}" <<'PY'
        import json
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(sys.argv[1]) / "scorer"))
        from compute_score import compute_score
        result = compute_score(Path("/tmp/output"), None, Path(sys.argv[2]))
        Path("/tmp/openfoam-test-logs/reward.json").write_text(json.dumps(result, indent=2))
        print("score:", result.get("score"))
        PY
        """
    )


GRADER_LIB = r'''from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "0/U",
    "0/p",
    "constant/transportProperties",
    "constant/fvOptions",
    "system/blockMeshDict",
    "system/controlDict",
    "system/fvSchemes",
    "system/fvSolution",
]


@dataclass
class RunnerResult:
    ok: bool
    mesh_ok: bool
    solver_ok: bool
    post_ok: bool
    metrics_ok: bool
    no_numerical_errors: bool
    metrics: dict[str, float]
    reason: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def submission_case(workspace: Path) -> tuple[Path | None, str]:
    folder = workspace / "fixed_case"
    if folder.is_dir():
        return folder, "folder"
    archive = workspace / "fixed_case.tar.gz"
    if not archive.exists():
        return None, "missing fixed_case.tar.gz"
    tmp = Path(tempfile.mkdtemp(prefix="synthetic-openfoam-submission-"))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            root = tmp.resolve()
            for member in tar.getmembers():
                target = (tmp / member.name).resolve()
                if os.path.commonpath([root, target]) != str(root):
                    raise ValueError(f"unsafe archive member: {member.name}")
            tar.extractall(tmp)
    except Exception as exc:  # noqa: BLE001 - grader must not crash on invalid submissions.
        shutil.rmtree(tmp, ignore_errors=True)
        return None, f"extract failed: {exc}"
    children = [p for p in tmp.iterdir() if p.is_dir()]
    return (children[0] if children else tmp), "archive"


def required_files_present(case_dir: Path) -> tuple[bool, list[str]]:
    missing = [rel for rel in REQUIRED_FILES if not (case_dir / rel).exists()]
    return not missing, missing


def forbidden_unchanged(case_dir: Path, forbidden: list[dict[str, str]]) -> tuple[bool, list[str]]:
    changed: list[str] = []
    for item in forbidden:
        path = case_dir / item["path"]
        if not path.exists() or sha256_file(path) != item["sha256"]:
            changed.append(item["path"])
    return not changed, changed


def replace_regex(path: Path, pattern: str, repl: str) -> None:
    if not path.exists():
        return
    text = path.read_text()
    path.write_text(re.sub(pattern, repl, text, count=1, flags=re.S))


def apply_hidden_variant(case_dir: Path, variant: dict[str, Any]) -> None:
    changes = variant.get("changes", {})
    if "viscosity" in changes:
        replace_regex(case_dir / "constant" / "transportProperties", r"nu\s+[-+0-9.eE]+;", f"nu {float(changes['viscosity']):.6f};")
    if "write_interval" in changes:
        replace_regex(case_dir / "system" / "controlDict", r"writeInterval\s+\d+;", f"writeInterval {int(changes['write_interval'])};")
    if "inlet_velocity" in changes:
        value = float(changes["inlet_velocity"])
        u = case_dir / "0" / "U"
        if u.exists():
            text = u.read_text()
            text = re.sub(r"internalField\s+uniform\s+\([^)]+\);", f"internalField uniform ({value:g} 0 0);", text, count=1)
            next_text = re.sub(r"inlet\s*\{\s*type\s+fixedValue;\s*value\s+uniform\s+\([^)]+\);\s*\}", f"inlet {{ type fixedValue; value uniform ({value:g} 0 0); }}", text, count=1)
            if next_text == text:
                next_text = re.sub(r"movingWall\s*\{\s*type\s+fixedValue;\s*value\s+uniform\s+\([^)]+\);\s*\}", f"movingWall {{ type fixedValue; value uniform ({value:g} 0 0); }}", text, count=1)
            text = next_text
            u.write_text(text)
    if "probe_location" in changes:
        loc = changes["probe_location"]
        replace_regex(case_dir / "system" / "controlDict", r"probeLocations\s*\(\([^)]+\)\);", f"probeLocations (({loc[0]} {loc[1]} {loc[2]}));")
    if "source_strength" in changes:
        replace_regex(case_dir / "constant" / "fvOptions", r"U\s+\(\([^)]+\)\s+0\);", f"U (({float(changes['source_strength']):.3f} 0 0) 0);")
    if "mesh_grading" in changes:
        replace_regex(case_dir / "system" / "blockMeshDict", r"simpleGrading\s+\([^)]+\)", f"simpleGrading ({float(changes['mesh_grading']):g} 1 1)")


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def source_strength(case_dir: Path) -> float | None:
    match = re.search(r"U\s+\(\(\s*([-+0-9.eE]+)\s+0\s+0\s*\)\s+0\);", read_text(case_dir / "constant" / "fvOptions"))
    return float(match.group(1)) if match else None


def mesh_grading(case_dir: Path) -> float | None:
    match = re.search(r"simpleGrading\s+\(\s*([-+0-9.eE]+)", read_text(case_dir / "system" / "blockMeshDict"))
    return float(match.group(1)) if match else None


def family_repair_ok(case_dir: Path, family: str, expected: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if family == "boundary_condition_repair":
        p = read_text(case_dir / "0" / "p")
        ok = bool(re.search(r"outlet\s*\{[^}]*type\s+fixedValue;[^}]*value\s+uniform\s+0;", p, re.S))
        return ok, {"outlet_pressure_fixed_value": ok}
    if family == "solver_convergence_repair":
        schemes = read_text(case_dir / "system" / "fvSchemes")
        ok = "div(phi,U) Gauss linearUpwind grad(U);" in schemes or "div(phi,U) Gauss upwind;" in schemes
        return ok, {"stable_div_phi_u_scheme": ok}
    if family == "mesh_quality_repair":
        grading = mesh_grading(case_dir)
        ok = grading is not None and grading <= 2.0
        return ok, {"simple_grading_x": grading}
    if family == "postprocessing_instrumentation":
        control = read_text(case_dir / "system" / "controlDict")
        present = {name: bool(re.search(rf"\b{name}\b", control)) for name in ["residuals", "probes", "fieldMinMax"]}
        return all(present.values()), present
    if family == "source_term_calibration":
        value = source_strength(case_dir)
        target = float(expected.get("source_target") or 0.2)
        tol = float(expected.get("tolerance") or 0.05)
        ok = value is not None and abs(value - target) <= tol
        return ok, {"source_strength": value, "target": target, "tolerance": tol}
    return False, {"unknown_family": family}


def application(case_dir: Path) -> str:
    match = re.search(r"application\s+([A-Za-z0-9_]+)\s*;", read_text(case_dir / "system" / "controlDict"))
    return match.group(1) if match else "simpleFoam"


def residual_max(log: str) -> float | None:
    values = [float(v) for v in re.findall(r"Final residual =\s*([-+0-9.eE]+)", log)]
    return max(values) if values else None


def continuity_max(log: str) -> float | None:
    values = [float(v) for v in re.findall(r"(?:global|local) =\s*([-+0-9.eE]+)", log)]
    return max(abs(v) for v in values) if values else None


def post_outputs_ok(case_dir: Path, family: str) -> bool:
    post = case_dir / "postProcessing"
    if family == "postprocessing_instrumentation":
        control = read_text(case_dir / "system" / "controlDict")
        return all(name in control for name in ["residuals", "probes", "fieldMinMax"])
    if family == "source_term_calibration":
        return "fieldAverage" in read_text(case_dir / "system" / "controlDict")
    return True


def strip_function_objects_for_solver(case_dir: Path) -> None:
    control = case_dir / "system" / "controlDict"
    text = read_text(control)
    if "functions" in text:
        control.write_text(re.sub(r"functions\s*\{.*\}\s*$", "functions {}\n", text, count=1, flags=re.S))


class RealOpenFoamRunner:
    def run(self, case_dir: Path, expected: dict[str, Any], variant: dict[str, Any]) -> RunnerResult:
        timeout = int(expected.get("timeout_sec", 120))
        env_snippet = "source /etc/solver-envs.d/openfoam.sh >/dev/null 2>&1"
        solver = application(case_dir)
        family = str(expected.get("family", ""))
        post_ok = post_outputs_ok(case_dir, family)
        strip_function_objects_for_solver(case_dir)
        cmds = [
            ("blockMesh", f"{env_snippet} && blockMesh -case {case_dir}"),
            ("checkMesh", f"{env_snippet} && checkMesh -case {case_dir}"),
            (solver, f"{env_snippet} && {solver} -case {case_dir}"),
        ]
        logs = []
        mesh_ok = False
        solver_ok = False
        for name, cmd in cmds:
            proc = subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, timeout=timeout, check=False)
            log = proc.stdout + proc.stderr
            logs.append(log)
            if name == "blockMesh" and proc.returncode != 0:
                return RunnerResult(False, False, False, False, False, "nan" not in log.lower(), {}, log[-2000:])
            if name == "checkMesh":
                mesh_ok = proc.returncode == 0 and "failed" not in log.lower()
                if not mesh_ok:
                    return RunnerResult(False, False, False, False, False, "nan" not in log.lower(), {}, log[-2000:])
            if name == solver:
                solver_ok = proc.returncode == 0
                if not solver_ok:
                    return RunnerResult(False, mesh_ok, False, False, False, "nan" not in log.lower(), {}, log[-2000:])
        joined = "\n".join(logs)
        lowered = joined.lower().replace("floating point exception trapping enabled", "")
        bad = any(token in lowered for token in ["nan", "floating point exception", "fatal error", "diverg"])
        residual = residual_max(joined)
        continuity = continuity_max(joined)
        residual_limit = float(expected.get("residual_max") or 1e-4)
        mass_limit = float(expected.get("mass_balance_max") or 2e-3)
        metrics_ok = (residual is None or residual <= residual_limit * 10) and (continuity is None or continuity <= mass_limit * 10)
        metrics = {
            "residual": residual if residual is not None else -1.0,
            "mass_balance": continuity if continuity is not None else -1.0,
            "source_strength": source_strength(case_dir) if source_strength(case_dir) is not None else -1.0,
            "mesh_grading": mesh_grading(case_dir) if mesh_grading(case_dir) is not None else -1.0,
        }
        return RunnerResult(not bad and mesh_ok and solver_ok and post_ok and metrics_ok, mesh_ok, solver_ok, post_ok, metrics_ok, not bad, metrics, "real-openfoam")


def select_runner() -> Any:
    return RealOpenFoamRunner()
'''


COMPUTE_SCORE = r'''from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from openfoam_grader_lib import (
    apply_hidden_variant,
    family_repair_ok,
    forbidden_unchanged,
    load_json,
    required_files_present,
    select_runner,
    submission_case,
)


def _criterion(results: list[dict[str, Any]], cid: str, weight: float, passed: bool, detail: Any = None) -> float:
    value = weight if passed else 0.0
    results.append({"id": cid, "weight": weight, "passed": bool(passed), "score": value, "detail": detail})
    return value


def _result(score: float, criteria: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    detailed_metadata = dict(metadata)
    detailed_metadata["criteria"] = criteria
    return {
        "score": round(score, 6),
        "subscores": {item["id"]: item["score"] for item in criteria},
        "weights": {item["id"]: item["weight"] for item in criteria},
        "metadata": detailed_metadata,
    }


def compute_score(workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path) -> dict[str, Any]:
    _ = trajectory
    workspace = Path(workspace)
    private = Path(private)
    criteria: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"runner": "real-openfoam"}

    hidden_variants = load_json(private / "hidden_variants.json", []) or []
    expected = load_json(private / "expected_metrics.json", {}) or {}
    forbidden = load_json(private / "forbidden_edits.json", []) or []
    family = str(expected.get("family", ""))

    case_dir, submission_reason = submission_case(workspace)
    score = _criterion(criteria, "submission_exists_and_extracts", 0.10, case_dir is not None, submission_reason)
    if case_dir is None:
        return _result(score, criteria, metadata)

    required_ok, missing = required_files_present(case_dir)
    score += _criterion(criteria, "required_openfoam_files", 0.10, required_ok, missing)
    forbidden_ok, changed = forbidden_unchanged(case_dir, forbidden)
    score += _criterion(criteria, "forbidden_edits_unchanged", 0.10, forbidden_ok, changed)
    repair_ok, repair_detail = family_repair_ok(case_dir, family, expected)
    score += _criterion(criteria, "family_specific_repair_present", 0.20, repair_ok, repair_detail)

    runner = select_runner()
    variant_results = []
    all_mesh = all_solver = all_post = all_metrics = all_clean = True
    variants_applied = bool(hidden_variants)
    for variant in hidden_variants if repair_ok else []:
        tmp = Path(tempfile.mkdtemp(prefix="synthetic-openfoam-variant-"))
        try:
            variant_case = tmp / "case"
            shutil.copytree(case_dir, variant_case)
            apply_hidden_variant(variant_case, variant)
            run = runner.run(variant_case, expected, variant)
            variant_results.append({
                "name": variant.get("name"),
                "ok": run.ok,
                "mesh_ok": run.mesh_ok,
                "solver_ok": run.solver_ok,
                "post_ok": run.post_ok,
                "metrics_ok": run.metrics_ok,
                "no_numerical_errors": run.no_numerical_errors,
                "metrics": run.metrics,
                "reason": run.reason,
            })
            all_mesh = all_mesh and run.mesh_ok
            all_solver = all_solver and run.solver_ok
            all_post = all_post and run.post_ok
            all_metrics = all_metrics and run.metrics_ok
            all_clean = all_clean and run.no_numerical_errors
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if not repair_ok:
        all_mesh = all_solver = all_post = all_metrics = all_clean = False

    metadata["hidden_variants"] = variant_results
    score += _criterion(criteria, "hidden_variants_applied", 0.05, variants_applied, [v.get("name") for v in hidden_variants])
    score += _criterion(criteria, "mesh_generation_and_checkmesh", 0.10, all_mesh, variant_results)
    score += _criterion(criteria, "solver_reaches_end_time", 0.15, all_solver, variant_results)
    score += _criterion(criteria, "no_nan_fpe_or_fatal_errors", 0.10, all_clean, variant_results)
    score += _criterion(criteria, "required_postprocessing_outputs", 0.10, all_post, expected.get("postprocessing_required", []))
    score += _criterion(criteria, "engineering_metrics_within_tolerance", 0.10, all_metrics, expected)

    score = max(0.0, min(1.0, score))
    return _result(score, criteria, metadata)
'''

