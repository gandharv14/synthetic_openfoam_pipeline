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
    metrics: dict[str, Any]
    reason: str = ""
    logs: dict[str, str] | None = None
    postprocessing: dict[str, Any] | None = None


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


def block_span(text: str, name: str) -> tuple[int, int, str] | None:
    match = re.search(r"\b" + re.escape(name) + r"\b\s*\{", text)
    if not match:
        return None
    start = match.start()
    pos = match.end() - 1
    depth = 0
    for idx in range(pos, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return start, idx + 1, text[pos + 1:idx]
    return None


def entry_map(snippet: str) -> dict[str, str]:
    body_match = re.search(r"\{(.*)\}", snippet, flags=re.S)
    body = body_match.group(1) if body_match else snippet
    return {key: value.strip() for key, value in re.findall(r"\b([A-Za-z_][\w]*)\s+([^;{}]+);", body)}


def patch_entries(path: Path, patch: str) -> dict[str, str]:
    boundary = block_span(read_text(path), "boundaryField")
    if boundary is None:
        return {}
    patch_block = block_span(boundary[2], patch)
    if patch_block is None:
        return {}
    return entry_map(patch_block[2])


def set_patch(path: Path, patch: str, entries: dict[str, str]) -> None:
    text = read_text(path)
    boundary = block_span(text, "boundaryField")
    if boundary is None:
        return
    patch_block = block_span(boundary[2], patch)
    if patch_block is None:
        return
    order = [key for key in ("type", "inletValue", "value") if key in entries]
    order.extend(key for key in entries if key not in order)
    line = f"    {patch} {{ " + " ".join(f"{key} {entries[key]};" for key in order) + " }"
    body = boundary[2]
    path.write_text(text[:boundary[0]] + "boundaryField\n{" + body[:patch_block[0]] + line + body[patch_block[1]:] + "}" + text[boundary[1]:])


def set_internal_field(path: Path, value: str) -> None:
    replace_regex(path, r"\binternalField\s+uniform\s+[^;]+;", f"internalField uniform {value};")


def set_entry(path: Path, key: str, value: str) -> None:
    replace_regex(path, r"\b" + re.escape(key) + r"\s+[^;]+;", f"{key} {value};")


def set_probe_location(path: Path, loc: list[float]) -> None:
    replace_regex(path, r"\bprobeLocations\s*\(\([^)]+\)\);", f"probeLocations (({loc[0]} {loc[1]} {loc[2]}));")


def set_source(path: Path, value: float) -> None:
    replace_regex(path, r"\bU\s+\(\([^)]+\)\s+0\);", f"U (({value:.3f} 0 0) 0);")


def set_grading(path: Path, value: float) -> None:
    replace_regex(path, r"\bsimpleGrading\s+\([^)]+\)", f"simpleGrading ({value:g} 1 1)")


def functions_present(case_dir: Path, names: list[str]) -> dict[str, bool]:
    block = block_span(read_text(case_dir / "system" / "controlDict"), "functions")
    body = block[2] if block else ""
    return {name: bool(re.search(rf"\b{name}\b", body)) for name in names}


def apply_hidden_variant(case_dir: Path, variant: dict[str, Any]) -> None:
    changes = variant.get("changes", {})
    if "viscosity" in changes:
        set_entry(case_dir / "constant" / "transportProperties", "nu", f"{float(changes['viscosity']):.6f}")
    if "write_interval" in changes:
        set_entry(case_dir / "system" / "controlDict", "writeInterval", str(int(changes["write_interval"])))
    if "inlet_velocity" in changes:
        value = float(changes["inlet_velocity"])
        u = case_dir / "0" / "U"
        if u.exists():
            vector = f"({value:g} 0 0)"
            set_internal_field(u, vector)
            if patch_entries(u, "inlet"):
                set_patch(u, "inlet", {"type": "fixedValue", "value": f"uniform {vector}"})
            elif patch_entries(u, "movingWall"):
                set_patch(u, "movingWall", {"type": "fixedValue", "value": f"uniform {vector}"})
    if "probe_location" in changes:
        set_probe_location(case_dir / "system" / "controlDict", changes["probe_location"])
    if "source_strength" in changes:
        set_source(case_dir / "constant" / "fvOptions", float(changes["source_strength"]))
    if "mesh_grading" in changes:
        set_grading(case_dir / "system" / "blockMeshDict", float(changes["mesh_grading"]))


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def source_strength(case_dir: Path) -> float | None:
    match = re.search(r"U\s+\(\(\s*([-+0-9.eE]+)\s+0\s+0\s*\)\s+0\);", read_text(case_dir / "constant" / "fvOptions"))
    return float(match.group(1)) if match else None


def mesh_grading(case_dir: Path) -> float | None:
    match = re.search(r"simpleGrading\s+\(\s*([-+0-9.eE]+)", read_text(case_dir / "system" / "blockMeshDict"))
    return float(match.group(1)) if match else None


BASELINE_BEHAVIOR_TARGETS: dict[str, dict[str, Any]] = {
    # Documented operating points used when measuring the submission as-is.
    # Hidden variants overwrite these axes afterwards, so the baseline run is
    # the only place where the submitted case's own calibration is observable.
    "flow_rate_targeting": {"inlet_velocity": 1.0},
    "transient_setup_repair": {"write_interval": 20},
}


def family_repair_ok(
    case_dir: Path,
    family: str,
    expected: dict[str, Any],
    runner: Any | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Behavior-based repair gate.

    Instead of string-matching dictionary contents, run the submitted case
    as-is through the OpenFOAM runner and require the measured quantities
    (flow rate, pressure drop, checkMesh quality, function-object outputs)
    to fall inside the ExpectedMetrics tolerance bands.
    """
    runner = runner if runner is not None else select_runner()
    variant = {"name": "baseline_behavior", "changes": dict(BASELINE_BEHAVIOR_TARGETS.get(family, {}))}
    tmp = Path(tempfile.mkdtemp(prefix="synthetic-openfoam-baseline-"))
    try:
        baseline_case = tmp / "case"
        shutil.copytree(case_dir, baseline_case)
        run = runner.run(baseline_case, expected, variant)
    except Exception as exc:  # noqa: BLE001 - grader must not crash on invalid submissions.
        return False, {"family": family, "mode": "behavioral-baseline-run", "error": str(exc)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return bool(run.ok), {
        "family": family,
        "mode": "behavioral-baseline-run",
        "baseline_targets": variant["changes"],
        "mesh_ok": run.mesh_ok,
        "solver_ok": run.solver_ok,
        "post_ok": run.post_ok,
        "metrics_ok": run.metrics_ok,
        "no_numerical_errors": run.no_numerical_errors,
        "reason": run.reason,
        "metrics": run.metrics,
        "postprocessing": run.postprocessing or {},
    }


def application(case_dir: Path) -> str:
    match = re.search(r"application\s+([A-Za-z0-9_]+)\s*;", read_text(case_dir / "system" / "controlDict"))
    return match.group(1) if match else "simpleFoam"


def residual_max(log: str) -> float | None:
    values = [float(v) for v in re.findall(r"Final residual =\s*([-+0-9.eE]+)", log)]
    return max(values) if values else None


def residual_window(log: str, limit: int = 20) -> dict[str, float]:
    values = [float(v) for v in re.findall(r"Final residual =\s*([-+0-9.eE]+)", log)]
    window = values[-limit:]
    return {
        "count": float(len(window)),
        "max": max(window) if window else -1.0,
        "last": window[-1] if window else -1.0,
    }


def continuity_max(log: str) -> float | None:
    values = [float(v) for v in re.findall(r"(?:global|local) =\s*([-+0-9.eE]+)", log)]
    return max(abs(v) for v in values) if values else None


def continuity_window(log: str, limit: int = 20) -> dict[str, float]:
    values = [abs(float(v)) for v in re.findall(r"(?:global|local) =\s*([-+0-9.eE]+)", log)]
    window = values[-limit:]
    return {
        "count": float(len(window)),
        "max": max(window) if window else -1.0,
        "last": window[-1] if window else -1.0,
    }


def numeric_values(text: str) -> list[float]:
    return [float(v) for v in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)]


def latest_time_dir(case_dir: Path) -> Path | None:
    times: list[tuple[float, Path]] = []
    for path in case_dir.iterdir() if case_dir.exists() else []:
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if value > 0:
            times.append((value, path))
    return max(times, default=(0.0, None))[1]


def positive_time_dirs(case_dir: Path) -> list[float]:
    times: list[float] = []
    for path in case_dir.iterdir() if case_dir.exists() else []:
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if value > 0:
            times.append(value)
    return sorted(times)


def _parse_foam_value(raw: str) -> list[float]:
    raw = raw.strip()
    if raw.startswith("(") and raw.endswith(")"):
        return numeric_values(raw)
    try:
        return [float(raw)]
    except ValueError:
        return []


def field_numeric_values(path: Path) -> list[float]:
    text = read_text(path)
    values: list[float] = []
    for match in re.finditer(r"(?:internalField|value)\s+uniform\s+(\([^;]+\)|[-+0-9.eE]+)\s*;", text):
        values.extend(_parse_foam_value(match.group(1)))
    for match in re.finditer(r"(?:internalField|value)\s+nonuniform[^{;]*\((.*?)\)\s*;", text, flags=re.S):
        values.extend(numeric_values(match.group(1)))
    return values


def latest_field_values(case_dir: Path, field: str) -> list[float]:
    latest = latest_time_dir(case_dir)
    if latest is not None and (latest / field).exists():
        values = field_numeric_values(latest / field)
        if values:
            return values
    return field_numeric_values(case_dir / "0" / field)


def patch_field_value(case_dir: Path, field: str, patch: str) -> float | None:
    """Measured value held on a boundary patch, preferring the solved state."""
    paths: list[Path] = []
    latest = latest_time_dir(case_dir)
    if latest is not None:
        paths.append(latest / field)
    paths.append(case_dir / "0" / field)
    for path in paths:
        if not path.exists():
            continue
        values = numeric_values(patch_entries(path, patch).get("value", ""))
        if values:
            return sum(values) / len(values)
    return None


def outlet_datum_band(expected: dict[str, Any]) -> float:
    # The outlet gauge pressure must sit near the documented zero datum; scale
    # the allowed band off the calibrated pressure-drop ceiling.
    return max(float(expected.get("tolerance") or 0.05) * max(float(expected.get("pressure_drop_max") or 5.0), 1.0), 0.05)


def table_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in read_text(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = numeric_values(stripped)
        if values:
            rows.append(values)
    return rows


def post_files(case_dir: Path, name: str) -> list[Path]:
    post = case_dir / "postProcessing"
    if not post.exists():
        return []
    return sorted(path for path in post.rglob("*") if path.is_file() and name.lower() in path.as_posix().lower())


def post_output_present(case_dir: Path, name: str) -> bool:
    aliases = {
        "residuals": ("residual",),
        "probes": ("probes", "probe"),
        "fieldMinMax": ("fieldminmax", "fieldMinMax"),
        "fieldAverage": ("fieldaverage", "fieldAverage"),
        "forceCoeffs": ("forcecoeffs", "forceCoeffs"),
    }
    wanted = aliases.get(name, (name,))
    return any(post_files(case_dir, item) for item in wanted)


def post_last_values(case_dir: Path, name: str) -> list[float]:
    rows: list[list[float]] = []
    for path in post_files(case_dir, name):
        rows.extend(table_rows(path))
    if not rows:
        return []
    # Exclude the first column, which is normally time.
    return rows[-1][1:] if len(rows[-1]) > 1 else rows[-1]


def post_pressure_drop(case_dir: Path) -> float | None:
    values = post_last_values(case_dir, "probes")
    if len(values) >= 2:
        return max(values) - min(values)
    p_values = latest_field_values(case_dir, "p")
    if len(p_values) >= 2:
        return max(p_values) - min(p_values)
    return None


def field_average_value(case_dir: Path, field: str) -> float | None:
    for key in ("fieldAverage", field):
        values = post_last_values(case_dir, key)
        if values:
            return values[0]
    values = latest_field_values(case_dir, field)
    return sum(values) / len(values) if values else None


def force_coefficients(case_dir: Path) -> dict[str, float]:
    for path in post_files(case_dir, "force"):
        rows = table_rows(path)
        if not rows:
            continue
        header = ""
        for line in read_text(path).splitlines():
            if line.strip().startswith("#"):
                header = line.strip("# \n")
        values = rows[-1]
        tokens = header.split()
        result: dict[str, float] = {}
        for name in ("Cd", "Cl"):
            if name in tokens:
                pos = tokens.index(name)
                if pos < len(values):
                    result[name] = values[pos]
        if "Cd" not in result and len(values) >= 3:
            result["Cd"] = values[2]
        if "Cl" not in result and len(values) >= 4:
            result["Cl"] = values[3]
        if result:
            return result
    return {}


def inlet_velocity(case_dir: Path) -> float | None:
    candidates: list[Path] = []
    latest = latest_time_dir(case_dir)
    if latest is not None:
        candidates.append(latest / "U")
    candidates.append(case_dir / "0" / "U")
    for path in candidates:
        match = re.search(r"inlet\s*\{[^}]*value\s+uniform\s+\(([-+0-9.eE]+)", read_text(path), re.S)
        if match:
            return float(match.group(1))
    values = latest_field_values(case_dir, "U")
    return values[0] if values else None


def block_mesh_cells(case_dir: Path) -> int | None:
    mesh = read_text(case_dir / "system" / "blockMeshDict")
    total = 0
    for nx, ny, nz in re.findall(r"\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s+simpleGrading", mesh):
        total += int(nx) * int(ny) * int(nz)
    return total or None


def inlet_patch_area(case_dir: Path) -> float:
    mesh = read_text(case_dir / "system" / "blockMeshDict")
    vertices_match = re.search(r"vertices\s*\((.*?)\);\s*blocks", mesh, re.S)
    inlet_match = re.search(r"inlet\s*\{[^}]*faces\s*\((.*?)\);\s*\}", mesh, re.S)
    if not vertices_match or not inlet_match:
        return 0.1
    coords = re.findall(r"\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)", vertices_match.group(1))
    vertices = [(float(x), float(y), float(z)) for x, y, z in coords]
    area = 0.0
    for face in re.findall(r"\(([\d\s]+)\)", inlet_match.group(1)):
        indices = [int(i) for i in face.split()]
        pts = [vertices[i] for i in indices if i < len(vertices)]
        if len(pts) < 3:
            continue
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        area += max(max(ys) - min(ys), 0.0) * max(max(zs) - min(zs), 0.0)
    return area or 0.1


def post_outputs_ok(case_dir: Path, family: str, expected: dict[str, Any], log: str = "") -> bool:
    return not missing_post_outputs(case_dir, family, expected, log)


def required_post_outputs(family: str, expected: dict[str, Any]) -> list[str]:
    required = list(expected.get("postprocessing_required") or [])
    if family == "force_coeff_instrumentation" and "forceCoeffs" not in required:
        required.append("forceCoeffs")
    if family == "source_term_calibration" and "fieldAverage" not in required:
        required.append("fieldAverage")
    return [name for name in required if name not in {"checkMesh", "fluxBalance"}]


def missing_post_outputs(case_dir: Path, family: str, expected: dict[str, Any], log: str = "") -> list[str]:
    missing: list[str] = []
    declared = functions_present(case_dir, required_post_outputs(family, expected))
    for name in required_post_outputs(family, expected):
        if post_output_present(case_dir, name):
            continue
        if name == "residuals" and residual_max(log) is not None:
            continue
        if declared.get(name):
            continue
        missing.append(name)
    return missing


def solver_completed(case_dir: Path, log: str, expected: dict[str, Any]) -> bool:
    latest = latest_time_dir(case_dir)
    end_time = expected.get("end_time", 100)
    try:
        expected_end = float(end_time)
    except (TypeError, ValueError):
        expected_end = 100.0
    if latest is not None:
        try:
            if float(latest.name) >= expected_end:
                return True
        except ValueError:
            pass
    lowered = log.lower()
    return "end" in lowered and "time =" in lowered


def anchor_metrics_ok(metrics: dict[str, Any], anchor: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    oracle = anchor.get("oracle") or {}
    baseline = anchor.get("baseline") or {}
    tolerances = anchor.get("tolerances") or {}
    if not oracle:
        return True, {"present": False}
    comparisons: dict[str, Any] = {"present": True, "checked": {}, "baseline_delta": {}}
    ok = True
    for key, target in oracle.items():
        if key not in metrics:
            continue
        observed = metrics[key]
        if not isinstance(observed, (int, float)) or observed < 0:
            continue
        target_value = float(target)
        tolerance = float(tolerances.get(key, 0.1))
        allowed = tolerance if tolerance >= 1 else max(abs(target_value) * tolerance, 1e-12)
        delta = abs(float(observed) - target_value)
        passed = delta <= allowed
        comparisons["checked"][key] = {
            "observed": float(observed),
            "target": target_value,
            "delta": delta,
            "allowed": allowed,
            "passed": passed,
        }
        ok = ok and passed
        if key in baseline and isinstance(baseline[key], (int, float)):
            denominator = abs(float(baseline[key]) - target_value)
            comparisons["baseline_delta"][key] = (delta / denominator) if denominator > 1e-12 else 0.0
    if not comparisons["checked"]:
        return False, comparisons
    return ok, comparisons


def physical_metrics_ok(case_dir: Path, expected: dict[str, Any], family: str, log: str, variant: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    residual = residual_max(log)
    continuity = continuity_max(log)
    residual_limit = float(expected.get("residual_max") or 1e-4)
    mass_limit = float(expected.get("mass_balance_max") or 2e-3)
    residual_ok = residual is None or residual <= residual_limit * 10
    continuity_ok = continuity is None or continuity <= mass_limit * 10
    metrics: dict[str, Any] = {
        "residual": residual if residual is not None else -1.0,
        "residual_window": residual_window(log),
        "mass_balance": continuity if continuity is not None else -1.0,
        "continuity_window": continuity_window(log),
        "source_strength": source_strength(case_dir) if source_strength(case_dir) is not None else -1.0,
        "mesh_grading": mesh_grading(case_dir) if mesh_grading(case_dir) is not None else -1.0,
        "mesh_cells": block_mesh_cells(case_dir) if block_mesh_cells(case_dir) is not None else -1,
    }
    checks = [residual_ok, continuity_ok]

    if family == "boundary_condition_repair":
        datum = patch_field_value(case_dir, "p", "outlet")
        metrics["outlet_pressure"] = datum if datum is not None else -1.0
        checks.append(datum is not None and abs(datum) <= outlet_datum_band(expected))
    elif family == "flow_rate_targeting":
        velocity = inlet_velocity(case_dir)
        area = inlet_patch_area(case_dir)
        target_velocity = float(variant.get("changes", {}).get("inlet_velocity", velocity if velocity is not None else 1.0))
        flow = velocity * area if velocity is not None else None
        target_flow = target_velocity * area
        tol = max(float(expected.get("tolerance") or 0.05), 0.05) * max(abs(target_flow), 1e-9)
        metrics.update({"flow_rate": flow if flow is not None else -1.0, "target_flow_rate": target_flow, "inlet_area": area})
        checks.append(flow is not None and abs(flow - target_flow) <= tol)
    elif family == "pressure_drop_calibration":
        pressure_drop = post_pressure_drop(case_dir)
        datum = patch_field_value(case_dir, "p", "outlet")
        metrics["pressure_drop"] = pressure_drop if pressure_drop is not None else -1.0
        metrics["outlet_pressure"] = datum if datum is not None else -1.0
        checks.append(
            pressure_drop is not None
            and float(expected.get("pressure_drop_min") or 0.0) <= pressure_drop <= float(expected.get("pressure_drop_max") or 1e9)
        )
        checks.append(datum is not None and abs(datum) <= outlet_datum_band(expected))
    elif family == "transient_setup_repair":
        times = positive_time_dirs(case_dir)
        end_time = float(expected.get("end_time") or 100)
        interval = float(variant.get("changes", {}).get("write_interval") or 20)
        expected_writes = max(int(end_time // max(interval, 1e-9)), 1)
        metrics.update({
            "write_count": float(len(times)),
            "expected_write_count": float(expected_writes),
            "latest_write_time": times[-1] if times else -1.0,
        })
        checks.append(len(times) >= expected_writes)
    elif family == "source_term_calibration":
        value = source_strength(case_dir)
        response = field_average_value(case_dir, "U")
        target = float(expected.get("source_target") or 0.2)
        tol = float(expected.get("tolerance") or 0.05)
        metrics.update({"source_response": response if response is not None else -1.0, "source_target": target})
        checks.append(value is not None and abs(value - target) <= tol)
        if response is not None:
            checks.append(response >= 0.0)
    elif family == "force_coeff_instrumentation":
        # Coefficients must come from real postProcessing output; never
        # substitute oracle anchor values for missing observations.
        coeffs = force_coefficients(case_dir)
        metrics.update({"force_Cd": coeffs.get("Cd", -1.0), "force_Cl": coeffs.get("Cl", -1.0)})
        checks.append("Cd" in coeffs and "Cl" in coeffs)
    elif family == "scalar_transport_repair":
        values = latest_field_values(case_dir, "T")
        avg = field_average_value(case_dir, "T")
        wall = patch_field_value(case_dir, "T", "movingWall")
        scalar_min = min(values) if values else avg
        scalar_max = max(values) if values else avg
        metrics.update({
            "scalar_average": avg if avg is not None else -1.0,
            "scalar_min": scalar_min if scalar_min is not None else -1.0,
            "scalar_max": scalar_max if scalar_max is not None else -1.0,
            "wall_scalar": wall if wall is not None else -1.0,
        })
        checks.append(scalar_min is not None and scalar_max is not None and 250 <= scalar_min <= scalar_max <= 400)
        # The heated wall must hold a measurably elevated temperature
        # (seed cases initialize the field at the 300 K ambient).
        checks.append(wall is not None and wall >= 305.0)
    elif family in {"mesh_resolution_tradeoff", "mesh_quality_repair"}:
        cells = block_mesh_cells(case_dir)
        grading = mesh_grading(case_dir)
        checks.append(cells is not None and cells <= int(expected.get("max_cells", 1000)))
        checks.append(grading is not None and grading <= 2.0)

    anchor_ok, anchor_detail = anchor_metrics_ok(metrics, expected.get("metric_anchor") or {})
    metrics["anchor_comparison"] = anchor_detail
    checks.append(anchor_ok)
    return all(checks), metrics


def strip_function_objects_for_solver(case_dir: Path) -> None:
    control = case_dir / "system" / "controlDict"
    text = read_text(control)
    if "functions" in text:
        control.write_text(re.sub(r"functions\s*\{.*\}\s*$", "functions {}\n", text, count=1, flags=re.S))


class RealOpenFoamRunner:
    def _run_command(self, name: str, command: str, timeout: int) -> tuple[bool, str, int]:
        proc = subprocess.run(["bash", "-lc", command], text=True, capture_output=True, timeout=timeout, check=False)
        log = proc.stdout + proc.stderr
        return proc.returncode == 0, log, proc.returncode

    def run(self, case_dir: Path, expected: dict[str, Any], variant: dict[str, Any]) -> RunnerResult:
        timeout = int(expected.get("timeout_sec", 120))
        env_snippet = "source /etc/solver-envs.d/openfoam.sh >/dev/null 2>&1"
        solver = application(case_dir)
        family = str(expected.get("family", ""))
        logs: dict[str, str] = {}
        mesh_ok = False
        solver_ok = False

        ok, log, _ = self._run_command("blockMesh", f"{env_snippet} && blockMesh -case {case_dir}", timeout)
        logs["blockMesh"] = log
        if not ok:
            return RunnerResult(False, False, False, False, False, "nan" not in log.lower(), {}, "blockMesh failed", logs, {"missing": []})

        ok, log, _ = self._run_command("checkMesh", f"{env_snippet} && checkMesh -case {case_dir}", timeout)
        logs["checkMesh"] = log
        mesh_ok = ok and "failed" not in log.lower()
        if not mesh_ok:
            return RunnerResult(False, False, False, False, False, "nan" not in log.lower(), {}, "checkMesh failed", logs, {"missing": []})

        control = case_dir / "system" / "controlDict"
        original_control = read_text(control)
        strip_function_objects_for_solver(case_dir)
        ok, log, _ = self._run_command(solver, f"{env_snippet} && {solver} -case {case_dir}", timeout)
        logs[solver] = log
        solver_ok = ok and solver_completed(case_dir, log, expected)
        if original_control:
            control.write_text(original_control)
        if not solver_ok:
            return RunnerResult(False, mesh_ok, False, False, False, "nan" not in log.lower(), {}, "solver did not reach expected end time", logs, {"missing": missing_post_outputs(case_dir, family, expected, log)})

        joined = "\n".join(logs.values())
        missing_post = missing_post_outputs(case_dir, family, expected, joined)
        if missing_post:
            ok, post_log, _ = self._run_command("postProcess", f"{env_snippet} && postProcess -case {case_dir} -latestTime", timeout)
            logs["postProcess"] = post_log
            joined = "\n".join(logs.values())
            missing_post = missing_post_outputs(case_dir, family, expected, joined)

        lowered = joined.lower().replace("floating point exception trapping enabled", "")
        bad = any(token in lowered for token in ["nan", "floating point exception", "fatal error", "diverg"])
        post_ok = not missing_post
        metrics_ok, metrics = physical_metrics_ok(case_dir, expected, family, joined, variant)
        postprocessing = {"required": required_post_outputs(family, expected), "missing": missing_post}
        return RunnerResult(not bad and mesh_ok and solver_ok and post_ok and metrics_ok, mesh_ok, solver_ok, post_ok, metrics_ok, not bad, metrics, "real-openfoam", logs, postprocessing)


def select_runner() -> Any:
    return RealOpenFoamRunner()
