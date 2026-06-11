from __future__ import annotations

import json
from textwrap import dedent

from synthetic_openfoam_pipeline.openfoam.case_model import (
    SCALAR_HOT_VALUE,
    SNAPPY_LOCATION_IN_MESH,
    SNAPPY_REFINEMENT_LEVEL,
    TRANSIENT_DELTA_T,
    TRANSIENT_END_TIME,
    TRANSIENT_PIMPLE_BLOCK,
    TURBULENCE_WALL_FUNCTIONS,
    turbulence_inlet_values,
    turbulence_second_field,
)


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


def solution_script(
    task_id: str,
    family: str,
    marker: dict,
    parameters: dict | None = None,
    families: list[str] | None = None,
    case_kind: str | None = None,
) -> str:
    _ = (task_id, marker)
    params = parameters or {}
    # The heated patch carrying the scalar BC follows the seed geometry.
    scalar_patch = "movingWall" if "cavity" in (case_kind or "cavity") else "walls"
    cells_x = int(params.get("cells_x", 20))
    cells_y = int(params.get("cells_y", 10))
    # Goal-seeking targets live in the seed parameters; repairs restore them.
    inlet_velocity = float(params.get("inlet_velocity", 1.0))
    outlet_pressure = float(params.get("outlet_pressure", 0.0))
    source_strength = float(params.get("source_strength", 0.2))
    # Turbulent RANS seeds: the consistent boundary state is derived from the
    # same hash-picked Reynolds grid used at seed-generation time.
    turbulence_model = str(params.get("turbulence_model", "kOmegaSST"))
    turbulence = turbulence_inlet_values(params)
    turbulence_second = turbulence_second_field(turbulence_model)
    turbulence_k = f"{turbulence['k']:g}"
    turbulence_second_value = f"{turbulence[turbulence_second]:g}"
    turbulence_second_wall = TURBULENCE_WALL_FUNCTIONS[turbulence_second]
    # Chain the primary repair with any difficulty-stacked secondary repairs.
    repair_families = list(dict.fromkeys([family, *(families or [])]))
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

        def replace_text(path: Path, old: str, new: str) -> None:
            text = path.read_text()
            if old not in text:
                raise SystemExit(f"expected text not found in {{path}}: {{old}}")
            path.write_text(text.replace(old, new, 1))

        def set_regex(path: Path, pattern: str, repl: str) -> None:
            text = path.read_text()
            next_text, count = re.subn(pattern, repl, text, count=1)
            if count != 1:
                raise SystemExit(f"pattern {{pattern}} not found in {{path}}")
            path.write_text(next_text)

        def apply_repair(family: str) -> None:
            if family == "boundary_condition_repair":
                set_patch(case / "0" / "p", "outlet", {{"type": "fixedValue", "value": "uniform {outlet_pressure:g}"}})
            elif family == "postprocessing_instrumentation":
                set_functions(case / "system" / "controlDict", FUNCTIONS)
            elif family == "solver_convergence_repair":
                set_div_phi_u(case / "system" / "fvSchemes", "Gauss linearUpwind grad(U)")
            elif family == "mesh_quality_repair":
                set_grading(case / "system" / "blockMeshDict", 1.5)
            elif family == "source_term_calibration":
                set_source(case / "constant" / "fvOptions", {source_strength})
            elif family == "flow_rate_targeting":
                set_patch(case / "0" / "U", "inlet", {{"type": "fixedValue", "value": "uniform ({inlet_velocity:g} 0 0)"}})
            elif family == "pressure_drop_calibration":
                set_patch(case / "0" / "p", "outlet", {{"type": "fixedValue", "value": "uniform {outlet_pressure:g}"}})
            elif family == "transient_setup_repair":
                set_entry(case / "system" / "controlDict", "deltaT", "{TRANSIENT_DELTA_T:g}")
                set_entry(case / "system" / "controlDict", "endTime", "{TRANSIENT_END_TIME:g}")
                set_regex(case / "system" / "fvSchemes", r"ddtSchemes\\s*\\{{[^}}]*\\}}", "ddtSchemes {{ default backward; }}")
                set_regex(case / "system" / "fvSolution", r"PIMPLE\\s*\\{{[^}}]*\\}}", "{TRANSIENT_PIMPLE_BLOCK}")
            elif family == "force_coeff_instrumentation":
                set_functions(case / "system" / "controlDict", FORCE_FUNCTIONS)
            elif family == "scalar_transport_repair":
                set_patch(case / "0" / "T", "{scalar_patch}", {{"type": "fixedValue", "value": "uniform {SCALAR_HOT_VALUE:g}"}})
            elif family == "mesh_resolution_tradeoff":
                set_cells(case / "system" / "blockMeshDict", ({cells_x * 4}, {cells_y * 4}, 1), ({cells_x}, {cells_y}, 1))
            elif family == "relaxation_factor_repair":
                replace_text(
                    case / "system" / "fvSolution",
                    "relaxationFactors {{ fields {{ p 0.95; }} equations {{ U 0.95; }} }}",
                    "relaxationFactors {{ fields {{ p 0.3; }} equations {{ U 0.7; }} }}",
                )
            elif family == "wall_patch_type_repair":
                replace_text(case / "0" / "U", "walls {{ type slip; }}", "walls {{ type noSlip; }}")
            elif family == "field_dimensions_repair":
                replace_text(case / "0" / "p", "dimensions [0 1 -2 0 0 0 0];", "dimensions [0 2 -2 0 0 0 0];")
            elif family == "residual_control_repair":
                replace_text(
                    case / "system" / "fvSolution",
                    "residualControl {{ p 0.5; U 0.5; }}",
                    "residualControl {{ p 1e-4; U 1e-4; }}",
                )
            elif family == "probe_placement_repair":
                replace_text(case / "system" / "controlDict", "probeLocations ((25 5 0.05));", "probeLocations ((1 0.5 0.05));")
            elif family == "snappy_hex_repair":
                set_regex(case / "system" / "snappyHexMeshDict", r"\\blevel\\s+\\(\\s*\\d+\\s+\\d+\\s*\\);", "{SNAPPY_REFINEMENT_LEVEL}")
                set_regex(case / "system" / "snappyHexMeshDict", r"\\blocationInMesh\\s+\\([^)]*\\);", "{SNAPPY_LOCATION_IN_MESH}")
            elif family == "turbulence_bc_repair":
                set_patch(case / "0" / "k", "walls", {{"type": "kqRWallFunction", "value": "uniform {turbulence_k}"}})
                set_patch(case / "0" / "k", "inlet", {{"type": "fixedValue", "value": "uniform {turbulence_k}"}})
                set_patch(case / "0" / "{turbulence_second}", "walls", {{"type": "{turbulence_second_wall}", "value": "uniform {turbulence_second_value}"}})
                set_patch(case / "0" / "{turbulence_second}", "inlet", {{"type": "fixedValue", "value": "uniform {turbulence_second_value}"}})
            else:
                raise SystemExit(f"unknown family: {{family}}")

        for family in {repair_families!r}:
            apply_repair(family)
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
        details = {
            "score": result.get("score"),
            "criteria": result.get("metadata", {}).get("criteria", []),
            "hidden_variants": result.get("metadata", {}).get("hidden_variants", []),
        }
        Path("/tmp/openfoam-test-logs/reward-details.json").write_text(json.dumps(details, indent=2))
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


def relaxation_factors(case_dir: Path) -> dict[str, float]:
    block = block_span(read_text(case_dir / "system" / "fvSolution"), "relaxationFactors")
    body = block[2] if block else ""
    return {key: float(value) for key, value in re.findall(r"\b(p|U)\s+([-+0-9.eE]+);", body)}


def control_dict_value(case_dir: Path, key: str) -> float | None:
    match = re.search(r"\b" + re.escape(key) + r"\s+([-+0-9.eE]+)\s*;", read_text(case_dir / "system" / "controlDict"))
    return float(match.group(1)) if match else None


def ddt_scheme(case_dir: Path) -> str | None:
    block = block_span(read_text(case_dir / "system" / "fvSchemes"), "ddtSchemes")
    if block is None:
        return None
    match = re.search(r"\bdefault\s+([A-Za-z0-9]+)", block[2])
    return match.group(1) if match else None


def pimple_settings(case_dir: Path) -> dict[str, float]:
    block = block_span(read_text(case_dir / "system" / "fvSolution"), "PIMPLE")
    body = block[2] if block else ""
    return {
        key: float(value)
        for key, value in re.findall(r"\b(nOuterCorrectors|nCorrectors|nNonOrthogonalCorrectors)\s+([-+0-9.eE]+);", body)
    }


def courant_max(log: str) -> float | None:
    values = [
        float(v)
        for v in re.findall(r"Courant Number mean[:=\s]+[-+0-9.eE]+\s+max[:=\s]+([-+0-9.eE]+)", log)
    ]
    return max(values) if values else None


def checkmesh_cell_count(log: str) -> int | None:
    """Cell count of the generated mesh as reported by checkMesh."""
    values = [int(v) for v in re.findall(r"\bcells:\s+(\d+)", log)]
    return values[-1] if values else None


def checkmesh_max_nonortho(log: str) -> float | None:
    values = [float(v) for v in re.findall(r"non-orthogonality Max:\s*([-+0-9.eE]+)", log)]
    return max(values) if values else None


def checkmesh_max_skewness(log: str) -> float | None:
    values = [float(v) for v in re.findall(r"[Mm]ax skewness = ([-+0-9.eE]+)", log)]
    return max(values) if values else None


def snappy_settings(case_dir: Path) -> dict[str, Any]:
    """Pinned snappyHexMesh controls parsed from the case dictionary."""
    text = read_text(case_dir / "system" / "snappyHexMeshDict")
    settings: dict[str, Any] = {}
    level = re.search(r"\blevel\s+\(\s*(\d+)\s+(\d+)\s*\)", text)
    if level:
        settings["refinement_level"] = max(int(level.group(1)), int(level.group(2)))
    location = re.search(r"\blocationInMesh\s+\(([^)]*)\)", text)
    if location:
        settings["location"] = [float(v) for v in location.group(1).split()]
    box_min = re.search(r"\bmin\s+\(([^)]*)\)", text)
    box_max = re.search(r"\bmax\s+\(([^)]*)\)", text)
    if box_min and box_max:
        settings["obstacle_box"] = (
            [float(v) for v in box_min.group(1).split()],
            [float(v) for v in box_max.group(1).split()],
        )
    return settings


def residual_controls(case_dir: Path) -> dict[str, float]:
    block = block_span(read_text(case_dir / "system" / "fvSolution"), "residualControl")
    body = block[2] if block else ""
    return {key: float(value) for key, value in re.findall(r"\b(p|U)\s+([-+0-9.eE]+);", body)}


def field_dimensions(case_dir: Path, field: str) -> list[float] | None:
    match = re.search(r"\bdimensions\s+\[([^\]]+)\]\s*;", read_text(case_dir / "0" / field))
    return [float(token) for token in match.group(1).split()] if match else None


def probe_point(case_dir: Path) -> list[float] | None:
    match = re.search(r"\bprobeLocations\s*\(\(([^)]+)\)\);", read_text(case_dir / "system" / "controlDict"))
    return [float(token) for token in match.group(1).split()] if match else None


def mesh_bounding_box(case_dir: Path) -> tuple[list[float], list[float]] | None:
    mesh = read_text(case_dir / "system" / "blockMeshDict")
    vertices_match = re.search(r"vertices\s*\((.*?)\);\s*blocks", mesh, re.S)
    if not vertices_match:
        return None
    coords = re.findall(r"\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)", vertices_match.group(1))
    if not coords:
        return None
    points = [(float(x), float(y), float(z)) for x, y, z in coords]
    lows = [min(point[axis] for point in points) for axis in range(3)]
    highs = [max(point[axis] for point in points) for axis in range(3)]
    return lows, highs


def baseline_behavior_changes(family: str, expected: dict[str, Any]) -> dict[str, Any]:
    """Operating-point targets used when measuring the submission as-is.

    Hidden variants overwrite these axes afterwards, so the baseline run is
    the only place where the submitted case's own calibration is observable.
    Goal-seeking targets come from the task's expected metrics.
    """
    changes: dict[str, Any] = {}
    if family == "flow_rate_targeting":
        changes["inlet_velocity"] = float(expected.get("target_inlet_velocity") or 1.0)
    elif family == "transient_setup_repair":
        changes["write_interval"] = 20
    return changes


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
    variant = {"name": "baseline_behavior", "changes": baseline_behavior_changes(family, expected)}
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


def turbulence_patch(case_dir: Path, field: str, patch: str) -> dict[str, str]:
    """Boundary entries for a turbulence field, preferring the submitted 0/ state."""
    initial = case_dir / "0" / field
    if initial.exists():
        entries = patch_entries(initial, patch)
        if entries:
            return entries
    latest = latest_time_dir(case_dir)
    if latest is not None and (latest / field).exists():
        return patch_entries(latest / field, patch)
    return {}


def first_numeric(text: str) -> float | None:
    values = numeric_values(text)
    return values[0] if values else None


def patch_field_mean(case_dir: Path, field: str, patch: str) -> float | None:
    """Mean of a patch's solved values, preferring the latest written state.

    Unlike `patch_field_value`, this handles nonuniform value lists (the
    leading list-size token is excluded), which written outflow patches such
    as inletOutlet carry after the solve.
    """
    paths: list[Path] = []
    latest = latest_time_dir(case_dir)
    if latest is not None:
        paths.append(latest / field)
    paths.append(case_dir / "0" / field)
    for path in paths:
        if not path.exists():
            continue
        raw = patch_entries(path, patch).get("value", "")
        if not raw:
            continue
        inner = re.search(r"\((.*)\)", raw, flags=re.S)
        values = numeric_values(inner.group(1)) if inner else numeric_values(raw)
        if values:
            return sum(values) / len(values)
    return None


def outlet_datum_target(expected: dict[str, Any]) -> float:
    return float(expected.get("outlet_pressure_target") or 0.0)


def outlet_datum_band(expected: dict[str, Any]) -> float:
    # The outlet gauge pressure must sit near the documented datum; scale the
    # allowed band off the calibrated pressure-drop ceiling.
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


def post_field_window_means(case_dir: Path, name: str, time_start: float = 0.0) -> dict[str, float]:
    """Per-field time-averaged post-processing signals over a fixed window.

    Rows from every matching table (e.g. probes/0/p, probes/0/U) whose time
    column falls inside [time_start, end] are averaged across all value
    columns, keyed by the file stem. Never returns instantaneous snapshots,
    so single-core reruns reproduce the same numbers.
    """
    samples: dict[str, list[float]] = {}
    for path in post_files(case_dir, name):
        for row in table_rows(path):
            if len(row) < 2 or row[0] < time_start:
                continue
            samples.setdefault(path.stem, []).extend(row[1:])
    return {stem: sum(values) / len(values) for stem, values in samples.items() if values}


def mean_force_coefficients(case_dir: Path, time_start: float = 0.0) -> dict[str, float]:
    """Time-averaged Cd/Cl over the [time_start, end] window."""
    for path in post_files(case_dir, "force"):
        rows = [row for row in table_rows(path) if row and row[0] >= time_start]
        if not rows:
            continue
        header = ""
        for line in read_text(path).splitlines():
            if line.strip().startswith("#"):
                header = line.strip("# \n")
        tokens = header.split()
        columns: dict[str, int] = {}
        for coeff in ("Cd", "Cl"):
            if coeff in tokens:
                columns[coeff] = tokens.index(coeff)
        if "Cd" not in columns and all(len(row) >= 3 for row in rows):
            columns["Cd"] = 2
        if "Cl" not in columns and all(len(row) >= 4 for row in rows):
            columns["Cl"] = 3
        result: dict[str, float] = {}
        for coeff, position in columns.items():
            values = [row[position] for row in rows if position < len(row)]
            if values:
                result[coeff] = sum(values) / len(values)
        if result:
            return result
    return {}


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
        checks.append(datum is not None and abs(datum - outlet_datum_target(expected)) <= outlet_datum_band(expected))
    elif family == "flow_rate_targeting":
        velocity = inlet_velocity(case_dir)
        area = inlet_patch_area(case_dir)
        target_velocity = variant.get("changes", {}).get("inlet_velocity")
        if target_velocity is None:
            target_velocity = expected.get("target_inlet_velocity")
        if target_velocity is None:
            target_velocity = velocity if velocity is not None else 1.0
        target_velocity = float(target_velocity)
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
        metrics["outlet_pressure_target"] = outlet_datum_target(expected)
        checks.append(
            pressure_drop is not None
            and float(expected.get("pressure_drop_min") or 0.0) <= pressure_drop <= float(expected.get("pressure_drop_max") or 1e9)
        )
        checks.append(datum is not None and abs(datum - outlet_datum_target(expected)) <= outlet_datum_band(expected))
    elif family == "transient_setup_repair":
        # Grade the transient integration setup plus TIME-AVERAGED quantities
        # over the fixed window; instantaneous snapshots are never compared.
        window = float(expected.get("average_window_start") or 0.0)
        end_time = float(expected.get("end_time") or 100.0)
        scheme = ddt_scheme(case_dir)
        dt = control_dict_value(case_dir, "deltaT")
        pimple = pimple_settings(case_dir)
        courant = courant_max(log)
        times = positive_time_dirs(case_dir)
        coeffs = mean_force_coefficients(case_dir, window)
        probe_means = post_field_window_means(case_dir, "probes", window)
        latest = latest_time_dir(case_dir)
        field_average_ok = latest is not None and (latest / "UMean").exists()
        metrics.update({
            "ddt_backward": 1.0 if scheme == "backward" else -1.0,
            "delta_t": dt if dt is not None else -1.0,
            "pimple_outer_correctors": pimple.get("nOuterCorrectors", -1.0),
            "pimple_correctors": pimple.get("nCorrectors", -1.0),
            "courant_max": courant if courant is not None else -1.0,
            "latest_write_time": times[-1] if times else -1.0,
            "mean_force_Cd": coeffs.get("Cd", -1.0),
            "mean_force_Cl_mag": abs(coeffs["Cl"]) if "Cl" in coeffs else -1.0,
            "mean_probe_p_mag": abs(probe_means["p"]) if "p" in probe_means else -1.0,
            "mean_probe_U_mag": abs(probe_means["U"]) if "U" in probe_means else -1.0,
            "field_average_written": 1.0 if field_average_ok else -1.0,
        })
        checks.append(scheme == "backward")
        max_dt = expected.get("max_delta_t")
        checks.append(dt is not None and (max_dt is None or dt <= float(max_dt) + 1e-12))
        checks.append(bool(pimple) and pimple.get("nOuterCorrectors", 0) >= 1 and pimple.get("nCorrectors", 0) >= 1)
        max_co = expected.get("max_courant")
        checks.append(courant is None or max_co is None or courant <= float(max_co))
        # Writes may not land exactly on endTime for every hidden write
        # cadence; require the run to reach the horizon within one interval.
        checks.append(bool(times) and times[-1] >= end_time - 0.6)
        checks.append("Cd" in coeffs and "Cl" in coeffs)
        checks.append("p" in probe_means and "U" in probe_means)
        checks.append(field_average_ok)
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
        # T is actively solved by the scalarTransport functionObject during
        # the run; grade transported outcomes from the solved field: outlet
        # mean temperature, energy-balance wall heat flux, and boundedness.
        field = str(expected.get("scalar_field") or "T")
        ambient = float(expected.get("scalar_ambient_value") or 300.0)
        hot = float(expected.get("scalar_hot_value") or 310.0)
        hot_patch = str(expected.get("scalar_hot_patch") or "movingWall")
        values = latest_field_values(case_dir, field)
        avg = field_average_value(case_dir, field)
        wall = patch_field_mean(case_dir, field, hot_patch)
        outlet_mean = patch_field_mean(case_dir, field, "outlet")
        scalar_min = min(values) if values else avg
        scalar_max = max(values) if values else avg
        latest = latest_time_dir(case_dir)
        # A transported solution is nonuniform at the solved time; the static
        # initial field stays uniform.
        transported = latest is not None and "nonuniform" in read_text(latest / field)
        heat_flux = None
        if outlet_mean is not None:
            velocity = inlet_velocity(case_dir)
            if velocity is not None:
                # Steady energy balance: wall heat input equals the convected
                # outflow, flow_rate * (T_out - T_in) (per rho*cp).
                heat_flux = velocity * inlet_patch_area(case_dir) * (outlet_mean - ambient)
        metrics.update({
            "scalar_average": avg if avg is not None else -1.0,
            "scalar_min": scalar_min if scalar_min is not None else -1.0,
            "scalar_max": scalar_max if scalar_max is not None else -1.0,
            "wall_scalar": wall if wall is not None else -1.0,
            "scalar_outlet_mean": outlet_mean if outlet_mean is not None else -1.0,
            "scalar_wall_heat_flux": heat_flux if heat_flux is not None else -1.0,
            "scalar_transported": 1.0 if transported else -1.0,
        })
        checks.append(transported)
        # Boundedness within the tolerance band around the boundary states.
        checks.append(
            scalar_min is not None
            and scalar_max is not None
            and ambient - 1.0 <= scalar_min <= scalar_max <= hot + 1.0
        )
        # The heated wall must hold a measurably elevated temperature.
        checks.append(wall is not None and wall >= ambient + 0.5 * (hot - ambient))
        checks.append(avg is not None and avg > ambient + 0.1)
        if outlet_mean is not None:
            # Channel seeds: the transported outlet mean must show real wall
            # heat input (positive flux) while staying below the wall value.
            checks.append(ambient + 0.2 <= outlet_mean <= hot + 1.0)
            checks.append(heat_flux is None or heat_flux > 0.0)
    elif family in {"mesh_resolution_tradeoff", "mesh_quality_repair"}:
        cells = block_mesh_cells(case_dir)
        grading = mesh_grading(case_dir)
        checks.append(cells is not None and cells <= int(expected.get("max_cells", 1000)))
        checks.append(grading is not None and grading <= 2.0)
    elif family == "relaxation_factor_repair":
        factors = relaxation_factors(case_dir)
        metrics.update({"relaxation_p": factors.get("p", -1.0), "relaxation_U": factors.get("U", -1.0)})
        # A stable steady SIMPLE strategy: heavy pressure under-relaxation,
        # bounded momentum relaxation; the residual checks above enforce the
        # measured convergence behavior.
        checks.append(0.0 < factors.get("p", -1.0) <= 0.5)
        checks.append(0.0 < factors.get("U", -1.0) <= 0.9)
    elif family == "wall_patch_type_repair":
        walls = patch_entries(case_dir / "0" / "U", "walls")
        wall_type = walls.get("type", "")
        viscous_wall = wall_type == "noSlip" or (wall_type == "fixedValue" and "(0 0 0)" in walls.get("value", ""))
        metrics["viscous_wall"] = 1.0 if viscous_wall else -1.0
        checks.append(viscous_wall)
    elif family == "field_dimensions_repair":
        dims = field_dimensions(case_dir, "p")
        metrics["pressure_dimensions_ok"] = 1.0 if dims == [0, 2, -2, 0, 0, 0, 0] else -1.0
        # Kinematic pressure for incompressible solvers: [0 2 -2 0 0 0 0].
        checks.append(dims == [0, 2, -2, 0, 0, 0, 0])
    elif family == "residual_control_repair":
        controls = residual_controls(case_dir)
        metrics.update({"residual_control_p": controls.get("p", -1.0), "residual_control_U": controls.get("U", -1.0)})
        # Convergence controls must be strict; the residual window above
        # verifies the run actually converged to that level.
        checks.append(bool(controls) and all(0.0 < value <= 1e-3 for value in controls.values()))
    elif family == "probe_placement_repair":
        probe = probe_point(case_dir)
        bounds = mesh_bounding_box(case_dir)
        inside = False
        if probe is not None and bounds is not None:
            lows, highs = bounds
            inside = all(lows[axis] <= probe[axis] <= highs[axis] for axis in range(3))
        metrics["probe_inside_domain"] = 1.0 if inside else -1.0
        checks.append(inside)
        checks.append(post_output_present(case_dir, "probes") or bool(functions_present(case_dir, ["probes"])["probes"]))
    elif family == "turbulence_bc_repair":
        model = str(expected.get("turbulence_model") or "kOmegaSST")
        second = "epsilon" if "epsilon" in model.lower() else "omega"
        wall_types = {"k": "kqRWallFunction", "omega": "omegaWallFunction", "epsilon": "epsilonWallFunction"}
        k_wall = turbulence_patch(case_dir, "k", "walls").get("type", "")
        second_wall = turbulence_patch(case_dir, second, "walls").get("type", "")
        wall_ok = k_wall == wall_types["k"] and second_wall == wall_types[second]
        inlet_k = first_numeric(turbulence_patch(case_dir, "k", "inlet").get("value", ""))
        inlet_second = first_numeric(turbulence_patch(case_dir, second, "inlet").get("value", ""))
        k_values = latest_field_values(case_dir, "k")
        k_mean = sum(k_values) / len(k_values) if k_values else None
        metrics.update({
            "turbulence_wall_functions": 1.0 if wall_ok else -1.0,
            "turbulence_inlet_k": inlet_k if inlet_k is not None else -1.0,
            f"turbulence_inlet_{second}": inlet_second if inlet_second is not None else -1.0,
            "turbulence_k_mean": k_mean if k_mean is not None else -1.0,
        })
        checks.append(wall_ok)
        # Freestream values must sit inside the tolerance band around the
        # Reynolds-derived inlet state held in the private expected metrics.
        target_k = expected.get("turbulence_inlet_k")
        if target_k:
            checks.append(inlet_k is not None and abs(inlet_k - float(target_k)) <= 0.10 * abs(float(target_k)))
        target_second = expected.get(f"turbulence_inlet_{second}")
        if target_second:
            checks.append(inlet_second is not None and abs(inlet_second - float(target_second)) <= 0.10 * abs(float(target_second)))
        checks.append(bool(k_values) and min(k_values) >= 0.0)
    elif family == "snappy_hex_repair":
        # Grade the GENERATED mesh through checkMesh quality bands plus
        # integrated quantities (obstacle forces, inlet flux); never per-cell
        # field comparisons.
        snappy = snappy_settings(case_dir)
        cells = checkmesh_cell_count(log)
        nonortho = checkmesh_max_nonortho(log)
        skewness = checkmesh_max_skewness(log)
        coeffs = force_coefficients(case_dir)
        velocity = inlet_velocity(case_dir)
        flow = velocity * inlet_patch_area(case_dir) if velocity is not None else None
        level = snappy.get("refinement_level")
        location = snappy.get("location")
        box = snappy.get("obstacle_box")
        location_in_flow = (
            location is not None
            and box is not None
            and len(location) == 3
            and not all(box[0][axis] <= location[axis] <= box[1][axis] for axis in range(3))
        )
        metrics.update({
            "mesh_cells_actual": float(cells) if cells is not None else -1.0,
            "mesh_nonortho_max": nonortho if nonortho is not None else -1.0,
            "mesh_skewness_max": skewness if skewness is not None else -1.0,
            "snappy_refinement_level": float(level) if level is not None else -1.0,
            "snappy_location_in_flow": 1.0 if location_in_flow else -1.0,
            "force_Cd": coeffs.get("Cd", -1.0),
            "force_Cl": coeffs.get("Cl", -1.0),
            "flow_rate": flow if flow is not None else -1.0,
        })
        # Pinned snappy controls: a sane refinement budget and a keep-point
        # in the flow region (outside the obstacle geometry).
        checks.append(level is not None and level <= 2)
        checks.append(location_in_flow)
        # checkMesh quality bands on the generated (snapped) mesh.
        cells_min = expected.get("mesh_cells_min")
        cells_max = expected.get("mesh_cells_max")
        checks.append(
            cells is not None
            and (cells_min is None or cells >= int(cells_min))
            and (cells_max is None or cells <= int(cells_max))
        )
        nonortho_limit = expected.get("max_nonortho")
        checks.append(nonortho is None or nonortho_limit is None or nonortho <= float(nonortho_limit))
        skewness_limit = expected.get("max_skewness")
        checks.append(skewness is None or skewness_limit is None or skewness <= float(skewness_limit))
        # Integrated quantities on the carved obstacle must exist; anchors
        # compare their values per hidden variant.
        checks.append("Cd" in coeffs and "Cl" in coeffs)
        checks.append(flow is not None and flow > 0.0)

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

        if (case_dir / "system" / "snappyHexMeshDict").exists():
            ok, log, _ = self._run_command(
                "snappyHexMesh", f"{env_snippet} && snappyHexMesh -overwrite -case {case_dir}", timeout
            )
            logs["snappyHexMesh"] = log
            if not ok:
                return RunnerResult(False, False, False, False, False, "nan" not in log.lower(), {}, "snappyHexMesh failed", logs, {"missing": []})

        ok, log, _ = self._run_command("checkMesh", f"{env_snippet} && checkMesh -case {case_dir}", timeout)
        logs["checkMesh"] = log
        mesh_ok = ok and "failed" not in log.lower()
        if not mesh_ok:
            return RunnerResult(False, False, False, False, False, "nan" not in log.lower(), {}, "checkMesh failed", logs, {"missing": []})

        # Time-averaged and transported-scalar grading need function objects
        # (probes/forces/fieldAverage/scalarTransport) running during the
        # solve; other steady cases keep the stripped-solve behavior.
        keep_functions = bool(expected.get("average_window_start") or expected.get("scalar_field"))
        control = case_dir / "system" / "controlDict"
        original_control = read_text(control)
        if not keep_functions:
            strip_function_objects_for_solver(case_dir)
        ok, log, _ = self._run_command(solver, f"{env_snippet} && {solver} -case {case_dir}", timeout)
        logs[solver] = log
        solver_ok = ok and solver_completed(case_dir, log, expected)
        if original_control and not keep_functions:
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

    hidden_payload = load_json(private / "hidden_variants.json", []) or []
    # Per-task derived grids ship as {"axes", "derived_grid", "variants"};
    # legacy payloads are a bare variant list.
    hidden_variants = hidden_payload.get("variants") or [] if isinstance(hidden_payload, dict) else hidden_payload
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

    runner = select_runner()
    # Behavior-based gate: the submission is run as-is and measured against the
    # ExpectedMetrics tolerance bands (no dictionary string matching).
    repair_ok, repair_detail = family_repair_ok(case_dir, family, expected, runner)
    score += _criterion(criteria, "family_specific_repair_present", 0.10, repair_ok, repair_detail)

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
                "log_tails": {name: text[-2000:] for name, text in (run.logs or {}).items()},
                "postprocessing": run.postprocessing or {},
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
    score += _criterion(criteria, "engineering_metrics_within_tolerance", 0.20, all_metrics, expected)

    score = max(0.0, min(1.0, score))
    return _result(score, criteria, metadata)
'''

