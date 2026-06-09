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

def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new))

family = 'solver_convergence_repair'
if family == "boundary_condition_repair":
    replace(case / "0" / "p", "outlet { type zeroGradient; }", "outlet { type fixedValue; value uniform 0; }")
elif family == "postprocessing_instrumentation":
    replace(case / "system" / "controlDict", "functions {}", FUNCTIONS)
elif family == "solver_convergence_repair":
    replace(case / "system" / "fvSchemes", "div(phi,U) Gauss linear;", "div(phi,U) Gauss linearUpwind grad(U);")
elif family == "mesh_quality_repair":
    replace(case / "system" / "blockMeshDict", "simpleGrading (20 1 1)", "simpleGrading (1.5 1 1)")
elif family == "source_term_calibration":
    replace(case / "constant" / "fvOptions", "U ((0.000 0 0) 0);", "U ((0.200 0 0) 0);")
else:
    raise SystemExit(f"unknown family: {family}")
PY
tar -czf "${OUT}/fixed_case.tar.gz" -C "${WORK}" case
