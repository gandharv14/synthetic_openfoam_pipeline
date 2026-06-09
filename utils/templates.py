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
        import re
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
        FORCE_FUNCTIONS = '''functions
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
            forceCoeffs
            {{
                type forceCoeffs;
                libs ("libforces.so");
                writeControl timeStep;
                writeInterval 1;
                patches (walls);
                rho rhoInf;
                rhoInf 1;
                CofR (0 0 0);
                liftDir (0 1 0);
                dragDir (1 0 0);
                pitchAxis (0 0 1);
                magUInf 1;
                lRef 1;
                Aref 1;
            }}
        }}
        '''

        def block_span(text: str, name: str):
            match = re.search(r"\\b" + re.escape(name) + r"\\b\\s*\\{{", text)
            if not match:
                return None
            start = match.start()
            pos = match.end() - 1
            depth = 0
            for idx in range(pos, len(text)):
                if text[idx] == "{{":
                    depth += 1
                elif text[idx] == "}}":
                    depth -= 1
                    if depth == 0:
                        return start, idx + 1, text[pos + 1:idx]
            raise SystemExit(f"unclosed OpenFOAM block {{name}}")

        def set_patch(path: Path, patch: str, entries: dict[str, str]) -> None:
            text = path.read_text()
            boundary = block_span(text, "boundaryField")
            if boundary is None:
                raise SystemExit(f"boundaryField not found in {{path}}")
            patch_block = block_span(boundary[2], patch)
            if patch_block is None:
                raise SystemExit(f"patch {{patch}} not found in {{path}}")
            order = [key for key in ("type", "inletValue", "value") if key in entries]
            order.extend(key for key in entries if key not in order)
            line = "    " + patch + " {{ " + " ".join(key + " " + entries[key] + ";" for key in order) + " }}"
            body = boundary[2]
            path.write_text(text[:boundary[0]] + "boundaryField\\n{{" + body[:patch_block[0]] + line + body[patch_block[1]:] + "}}" + text[boundary[1]:])

        def set_entry(path: Path, key: str, value: str) -> None:
            text = path.read_text()
            next_text, count = re.subn(r"(\\b" + re.escape(key) + r"\\s+)([^;]+);", r"\\g<1>" + value + ";", text, count=1)
            if count != 1:
                raise SystemExit(f"entry {{key}} not found in {{path}}")
            path.write_text(next_text)

        def set_functions(path: Path, functions: str) -> None:
            text = path.read_text()
            block = block_span(text, "functions")
            if block is None:
                path.write_text(text.rstrip() + "\\n" + functions)
            else:
                path.write_text(text[:block[0]] + functions.rstrip() + "\\n" + text[block[1]:])

        def set_div_phi_u(path: Path, scheme: str) -> None:
            text = path.read_text()
            next_text, count = re.subn(r"\\bdiv\\(phi,U\\)\\s+[^;]+;", "div(phi,U) " + scheme + ";", text, count=1)
            if count != 1:
                raise SystemExit(f"div(phi,U) not found in {{path}}")
            path.write_text(next_text)

        def set_source(path: Path, value: float) -> None:
            text = path.read_text()
            next_text, count = re.subn(r"\\bU\\s+\\(\\([^)]+\\)\\s+0\\);", "U ((%.3f 0 0) 0);" % value, text, count=1)
            if count != 1:
                raise SystemExit(f"source U entry not found in {{path}}")
            path.write_text(next_text)

        def set_grading(path: Path, x: float) -> None:
            text = path.read_text()
            next_text, count = re.subn(r"\\bsimpleGrading\\s+\\([^)]+\\)", "simpleGrading (%g 1 1)" % x, text)
            if count < 1:
                raise SystemExit(f"simpleGrading not found in {{path}}")
            path.write_text(next_text)

        def set_cells(path: Path, old: tuple[int, int, int], new: tuple[int, int, int]) -> None:
            text = path.read_text()
            old_text = "(%d %d %d) simpleGrading" % old
            new_text = "(%d %d %d) simpleGrading" % new
            if old_text not in text:
                raise SystemExit(f"cell count {{old_text}} not found in {{path}}")
            path.write_text(text.replace(old_text, new_text, 1))

        family = {family!r}
        if family == "boundary_condition_repair":
            set_patch(case / "0" / "p", "outlet", {{"type": "fixedValue", "value": "uniform 0"}})
        elif family == "postprocessing_instrumentation":
            set_functions(case / "system" / "controlDict", FUNCTIONS)
        elif family == "solver_convergence_repair":
            set_div_phi_u(case / "system" / "fvSchemes", "Gauss linearUpwind grad(U)")
        elif family == "mesh_quality_repair":
            set_grading(case / "system" / "blockMeshDict", 1.5)
        elif family == "source_term_calibration":
            set_source(case / "constant" / "fvOptions", 0.2)
        elif family == "flow_rate_targeting":
            set_patch(case / "0" / "U", "inlet", {{"type": "fixedValue", "value": "uniform (1 0 0)"}})
        elif family == "pressure_drop_calibration":
            set_patch(case / "0" / "p", "outlet", {{"type": "fixedValue", "value": "uniform 0"}})
        elif family == "transient_setup_repair":
            set_entry(case / "system" / "controlDict", "endTime", "100")
            set_entry(case / "system" / "controlDict", "writeInterval", "20")
        elif family == "force_coeff_instrumentation":
            set_functions(case / "system" / "controlDict", FORCE_FUNCTIONS)
        elif family == "scalar_transport_repair":
            set_patch(case / "0" / "T", "movingWall", {{"type": "fixedValue", "value": "uniform 310"}})
        elif family == "mesh_resolution_tradeoff":
            set_cells(case / "system" / "blockMeshDict", (80, 40, 1), (20, 10, 1))
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
    metrics: dict[str, Any]
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


def family_repair_ok(case_dir: Path, family: str, expected: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if family == "boundary_condition_repair":
        outlet = patch_entries(case_dir / "0" / "p", "outlet")
        ok = outlet.get("type") == "fixedValue" and outlet.get("value") == "uniform 0"
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
        present = functions_present(case_dir, ["residuals", "probes", "fieldMinMax"])
        return all(present.values()), present
    if family == "source_term_calibration":
        value = source_strength(case_dir)
        target = float(expected.get("source_target") or 0.2)
        tol = float(expected.get("tolerance") or 0.05)
        ok = value is not None and abs(value - target) <= tol
        return ok, {"source_strength": value, "target": target, "tolerance": tol}
    if family == "flow_rate_targeting":
        inlet = patch_entries(case_dir / "0" / "U", "inlet")
        ok = inlet.get("type") == "fixedValue" and inlet.get("value") == "uniform (1 0 0)"
        return ok, {"target_inlet_velocity": ok}
    if family == "pressure_drop_calibration":
        outlet = patch_entries(case_dir / "0" / "p", "outlet")
        ok = outlet.get("type") == "fixedValue" and outlet.get("value") == "uniform 0"
        return ok, {"zero_outlet_pressure_datum": ok}
    if family == "transient_setup_repair":
        control = read_text(case_dir / "system" / "controlDict")
        end_ok = bool(re.search(r"\bendTime\s+100;", control))
        write_ok = bool(re.search(r"\bwriteInterval\s+20;", control))
        return end_ok and write_ok, {"end_time_100": end_ok, "write_interval_20": write_ok}
    if family == "force_coeff_instrumentation":
        present = functions_present(case_dir, ["residuals", "probes", "fieldMinMax", "forceCoeffs"])
        return all(present.values()), present
    if family == "scalar_transport_repair":
        wall = patch_entries(case_dir / "0" / "T", "movingWall")
        ok = wall.get("type") == "fixedValue" and wall.get("value") == "uniform 310"
        return ok, {"heated_wall_temperature_fixed": ok}
    if family == "mesh_resolution_tradeoff":
        mesh = read_text(case_dir / "system" / "blockMeshDict")
        ok = "(20 10 1) simpleGrading (1.5 1 1)" in mesh and "(80 40 1)" not in mesh
        return ok, {"compact_cell_count": ok}
    return False, {"unknown_family": family}


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
    u = read_text(case_dir / "0" / "U")
    match = re.search(r"inlet\s*\{[^}]*value\s+uniform\s+\(([-+0-9.eE]+)", u, re.S)
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


def post_outputs_ok(case_dir: Path, family: str, expected: dict[str, Any]) -> bool:
    required = list(expected.get("postprocessing_required") or [])
    if family == "force_coeff_instrumentation" and "forceCoeffs" not in required:
        required.append("forceCoeffs")
    if family == "source_term_calibration" and "fieldAverage" not in required:
        required.append("fieldAverage")
    material = [name for name in required if name not in {"checkMesh", "fluxBalance"}]
    return all(post_output_present(case_dir, name) for name in material)


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

    if family == "flow_rate_targeting":
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
        metrics["pressure_drop"] = pressure_drop if pressure_drop is not None else -1.0
        checks.append(
            pressure_drop is not None
            and float(expected.get("pressure_drop_min") or 0.0) <= pressure_drop <= float(expected.get("pressure_drop_max") or 1e9)
        )
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
        coeffs = force_coefficients(case_dir)
        metrics.update({"force_Cd": coeffs.get("Cd", -1.0), "force_Cl": coeffs.get("Cl", -1.0)})
        checks.append("Cd" in coeffs and "Cl" in coeffs)
    elif family == "scalar_transport_repair":
        values = latest_field_values(case_dir, "T")
        avg = field_average_value(case_dir, "T")
        scalar_min = min(values) if values else avg
        scalar_max = max(values) if values else avg
        metrics.update({
            "scalar_average": avg if avg is not None else -1.0,
            "scalar_min": scalar_min if scalar_min is not None else -1.0,
            "scalar_max": scalar_max if scalar_max is not None else -1.0,
        })
        checks.append(scalar_min is not None and scalar_max is not None and 250 <= scalar_min <= scalar_max <= 400)
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
    def run(self, case_dir: Path, expected: dict[str, Any], variant: dict[str, Any]) -> RunnerResult:
        timeout = int(expected.get("timeout_sec", 120))
        env_snippet = "source /etc/solver-envs.d/openfoam.sh >/dev/null 2>&1"
        solver = application(case_dir)
        family = str(expected.get("family", ""))
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
        post_ok = post_outputs_ok(case_dir, family, expected)
        metrics_ok, metrics = physical_metrics_ok(case_dir, expected, family, joined, variant)
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
    metric_anchors = load_json(private / "metric_anchors.json", {}) or {}
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
            variant_expected = dict(expected)
            variant_expected.update(variant.get("expected_overrides") or {})
            anchor = (metric_anchors.get("by_variant") or {}).get(str(variant.get("name")), {})
            if anchor:
                variant_expected["metric_anchor"] = anchor
            run = runner.run(variant_case, variant_expected, variant)
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
                "expected_overrides": variant.get("expected_overrides") or {},
                "metric_anchor_present": bool(anchor),
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

