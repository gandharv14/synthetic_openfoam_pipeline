from __future__ import annotations

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
