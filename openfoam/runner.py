from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
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
    failed_stage: str = ""
    logs: dict[str, str] = field(default_factory=dict)


class OpenFoamRunner:
    @staticmethod
    def application(case_dir: Path) -> str:
        """Solver named in the case's controlDict (not hardcoded)."""
        control = case_dir / "system" / "controlDict"
        text = control.read_text() if control.exists() else ""
        match = re.search(r"\bapplication\s+([A-Za-z0-9_]+)\s*;", text)
        return match.group(1) if match else "simpleFoam"

    def run_case(self, case_dir: Path, timeout: int = 120) -> OpenFoamRunResult:
        if shutil.which("blockMesh") is None:
            return OpenFoamRunResult(False, False, False, False, False, "blockMesh not available", "env")
        # Source the solver env when present, but do not fail if it is missing
        # (host installs put the tools directly on PATH).
        env = "source /etc/solver-envs.d/openfoam.sh >/dev/null 2>&1"
        logs: dict[str, str] = {}
        stages = [("mesh", "blockMesh")]
        if (case_dir / "system" / "snappyHexMeshDict").exists():
            stages.append(("snappy", "snappyHexMesh -overwrite"))
        stages.extend([("checkMesh", "checkMesh"), ("solve", self.application(case_dir))])
        for name, tool in stages:
            try:
                result = run_command(["bash", "-lc", f"{env}; {tool} -case {case_dir}"], timeout=timeout)
            except subprocess.TimeoutExpired:
                logs[name] = f"[{tool} timed out after {timeout}s]"
                return OpenFoamRunResult(False, name != "mesh", False, False, False, f"{tool} timed out", name, logs)
            logs[name] = result.stdout + result.stderr
            if not result.ok:
                return OpenFoamRunResult(False, name != "mesh", False, False, False, result.stderr[-500:], name, logs)
        return OpenFoamRunResult(True, True, True, True, True, "real OpenFOAM run completed", "", logs)
