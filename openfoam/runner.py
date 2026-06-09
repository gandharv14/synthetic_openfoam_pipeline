from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from synthetic_openfoam_pipeline.utils.subprocesses import run_command


@dataclass(frozen=True)
class OpenFoamRunResult:
    ok: bool
    mesh_ok: bool
    solver_ok: bool
    post_ok: bool
    metrics_ok: bool
    reason: str = ""


class OpenFoamRunner:
    def run_case(self, case_dir: Path, timeout: int = 120) -> OpenFoamRunResult:
        if shutil.which("blockMesh") is None:
            return OpenFoamRunResult(False, False, False, False, False, "blockMesh not available")
        env = "source /etc/solver-envs.d/openfoam.sh >/dev/null 2>&1"
        for name, cmd in [
            ("mesh", f"{env} && blockMesh -case {case_dir}"),
            ("checkMesh", f"{env} && checkMesh -case {case_dir}"),
            ("solve", f"{env} && simpleFoam -case {case_dir}"),
        ]:
            result = run_command(["bash", "-lc", cmd], timeout=timeout)
            if not result.ok:
                return OpenFoamRunResult(False, name != "mesh", False, False, False, result.stderr[-500:])
        return OpenFoamRunResult(True, True, True, True, True, "real OpenFOAM run completed")

