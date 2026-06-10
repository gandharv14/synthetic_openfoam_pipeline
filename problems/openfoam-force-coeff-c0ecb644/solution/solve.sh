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
python3 - "${WORK}/case" <<'PY'
import re
import sys
from pathlib import Path
case = Path(sys.argv[1])

FUNCTIONS = '''functions
{
    residuals
    {
        type residuals;
        libs ("libutilityFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
    }
    probes
    {
        type probes;
        libs ("libsampling.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
        probeLocations ((1 0.5 0.05));
    }
    fieldMinMax
    {
        type fieldMinMax;
        libs ("libfieldFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
    }
}
'''
FORCE_FUNCTIONS = '''functions
{
    residuals
    {
        type residuals;
        libs ("libutilityFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
    }
    probes
    {
        type probes;
        libs ("libsampling.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
        probeLocations ((1 0.5 0.05));
    }
    fieldMinMax
    {
        type fieldMinMax;
        libs ("libfieldFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
    }
    forceCoeffs
    {
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
    }
}
'''

def block_span(text: str, name: str):
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
    raise SystemExit(f"unclosed OpenFOAM block {name}")

def set_patch(path: Path, patch: str, entries: dict[str, str]) -> None:
    text = path.read_text()
    boundary = block_span(text, "boundaryField")
    if boundary is None:
        raise SystemExit(f"boundaryField not found in {path}")
    patch_block = block_span(boundary[2], patch)
    if patch_block is None:
        raise SystemExit(f"patch {patch} not found in {path}")
    order = [key for key in ("type", "inletValue", "value") if key in entries]
    order.extend(key for key in entries if key not in order)
    line = "    " + patch + " { " + " ".join(key + " " + entries[key] + ";" for key in order) + " }"
    body = boundary[2]
    path.write_text(text[:boundary[0]] + "boundaryField\n{" + body[:patch_block[0]] + line + body[patch_block[1]:] + "}" + text[boundary[1]:])

def set_entry(path: Path, key: str, value: str) -> None:
    text = path.read_text()
    next_text, count = re.subn(r"(\b" + re.escape(key) + r"\s+)([^;]+);", r"\g<1>" + value + ";", text, count=1)
    if count != 1:
        raise SystemExit(f"entry {key} not found in {path}")
    path.write_text(next_text)

def set_functions(path: Path, functions: str) -> None:
    text = path.read_text()
    block = block_span(text, "functions")
    if block is None:
        path.write_text(text.rstrip() + "\n" + functions)
    else:
        path.write_text(text[:block[0]] + functions.rstrip() + "\n" + text[block[1]:])

def set_div_phi_u(path: Path, scheme: str) -> None:
    text = path.read_text()
    next_text, count = re.subn(r"\bdiv\(phi,U\)\s+[^;]+;", "div(phi,U) " + scheme + ";", text, count=1)
    if count != 1:
        raise SystemExit(f"div(phi,U) not found in {path}")
    path.write_text(next_text)

def set_source(path: Path, value: float) -> None:
    text = path.read_text()
    next_text, count = re.subn(r"\bU\s+\(\([^)]+\)\s+0\);", "U ((%.3f 0 0) 0);" % value, text, count=1)
    if count != 1:
        raise SystemExit(f"source U entry not found in {path}")
    path.write_text(next_text)

def set_grading(path: Path, x: float) -> None:
    text = path.read_text()
    next_text, count = re.subn(r"\bsimpleGrading\s+\([^)]+\)", "simpleGrading (%g 1 1)" % x, text)
    if count < 1:
        raise SystemExit(f"simpleGrading not found in {path}")
    path.write_text(next_text)

def set_cells(path: Path, old: tuple[int, int, int], new: tuple[int, int, int]) -> None:
    text = path.read_text()
    old_text = "(%d %d %d) simpleGrading" % old
    new_text = "(%d %d %d) simpleGrading" % new
    if old_text not in text:
        raise SystemExit(f"cell count {old_text} not found in {path}")
    path.write_text(text.replace(old_text, new_text, 1))

family = 'force_coeff_instrumentation'
if family == "boundary_condition_repair":
    set_patch(case / "0" / "p", "outlet", {"type": "fixedValue", "value": "uniform 0"})
elif family == "postprocessing_instrumentation":
    set_functions(case / "system" / "controlDict", FUNCTIONS)
elif family == "solver_convergence_repair":
    set_div_phi_u(case / "system" / "fvSchemes", "Gauss linearUpwind grad(U)")
elif family == "mesh_quality_repair":
    set_grading(case / "system" / "blockMeshDict", 1.5)
elif family == "source_term_calibration":
    set_source(case / "constant" / "fvOptions", 0.2)
elif family == "flow_rate_targeting":
    set_patch(case / "0" / "U", "inlet", {"type": "fixedValue", "value": "uniform (1 0 0)"})
elif family == "pressure_drop_calibration":
    set_patch(case / "0" / "p", "outlet", {"type": "fixedValue", "value": "uniform 0"})
elif family == "transient_setup_repair":
    set_entry(case / "system" / "controlDict", "endTime", "100")
    set_entry(case / "system" / "controlDict", "writeInterval", "20")
elif family == "force_coeff_instrumentation":
    set_functions(case / "system" / "controlDict", FORCE_FUNCTIONS)
elif family == "scalar_transport_repair":
    set_patch(case / "0" / "T", "movingWall", {"type": "fixedValue", "value": "uniform 310"})
elif family == "mesh_resolution_tradeoff":
    set_cells(case / "system" / "blockMeshDict", (80, 40, 1), (20, 10, 1))
else:
    raise SystemExit(f"unknown family: {family}")
PY
tar -czf "${OUT}/fixed_case.tar.gz" -C "${WORK}" case
