from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from synthetic_openfoam_pipeline.schemas import HiddenVariant, TaskCandidate
from synthetic_openfoam_pipeline.openfoam.case_model import case_geometry_stats
from synthetic_openfoam_pipeline.openfoam.dictionary_edit import apply_structured_edit, replace_required
from synthetic_openfoam_pipeline.utils.templates import GRADER_LIB


class AnchorPrecomputeAgent:
    def __init__(self, *, real_precompute: bool | None = None) -> None:
        self.real_precompute = (
            os.environ.get("SYNTHETIC_OPENFOAM_REAL_ANCHORS") == "1"
            if real_precompute is None
            else real_precompute
        )

    def attach_anchors(self, candidate: TaskCandidate) -> TaskCandidate:
        mode = "openfoam-precomputed" if self._can_precompute_real() else "deterministic-formula"
        anchors = {
            "version": 1,
            "family": candidate.family,
            "seed_case": candidate.seed_case.seed_id,
            "mode": mode,
            "by_variant": {
                variant.name: self._variant_anchor(candidate, variant)
                for variant in candidate.hidden_variants
            },
        }
        candidate.metric_anchors = anchors
        candidate.generation_report["metric_anchor_summary"] = {
            "version": anchors["version"],
            "mode": anchors["mode"],
            "variant_count": len(anchors["by_variant"]),
        }
        return candidate

    def _can_precompute_real(self) -> bool:
        return self.real_precompute and all(shutil.which(command) for command in ("blockMesh", "checkMesh", "simpleFoam"))

    def _variant_anchor(self, candidate: TaskCandidate, variant: HiddenVariant) -> dict[str, Any]:
        changes = variant.changes
        expected = asdict(candidate.expected_metrics)
        velocity = float(changes.get("inlet_velocity", candidate.seed_case.parameters.get("inlet_velocity", 1.0)))
        viscosity = float(changes.get("viscosity", candidate.seed_case.parameters.get("viscosity", 0.01)))
        source = float(changes.get("source_strength", expected.get("source_target") or candidate.seed_case.parameters.get("source_strength", 0.2)))
        grading = float(changes.get("mesh_grading", 1.5))
        write_interval = int(changes.get("write_interval", 20))
        stats = case_geometry_stats(candidate.seed_case)
        area = stats["inlet_area"]
        flow_rate = velocity * area
        pressure_drop = self._pressure_drop_anchor(velocity, viscosity, candidate.seed_case.case_kind)
        outlet_pressure = float(expected.get("outlet_pressure_target") or 0.0)
        mesh_cells = int(stats["mesh_cells"])
        scalar_average = 305.0 if "cavity" in candidate.seed_case.case_kind else 300.0
        force_cd = 0.35 + 0.05 * velocity
        force_cl = 0.02

        oracle = {
            "flow_rate": round(flow_rate, 6),
            "pressure_drop": round(pressure_drop, 6),
            "outlet_pressure": round(outlet_pressure, 6),
            "source_strength": round(source, 6),
            "source_response": round(max(source, 0.0), 6),
            "mesh_cells": mesh_cells,
            "mesh_grading": round(grading, 6),
            "force_Cd": round(force_cd, 6),
            "force_Cl": round(force_cl, 6),
            "scalar_average": round(scalar_average, 6),
            "write_interval": write_interval,
        }
        baseline = {
            "flow_rate": round(flow_rate * 0.35, 6),
            "pressure_drop": round(pressure_drop * 1.4, 6),
            "outlet_pressure": round(outlet_pressure + 2.0, 6),
            "source_strength": 0.0,
            "source_response": 0.0,
            "mesh_cells": max(mesh_cells * 4, mesh_cells),
            "mesh_grading": round(max(grading * 10, 20.0), 6),
            "force_Cd": round(force_cd * 1.25, 6),
            "force_Cl": round(force_cl * 1.5, 6),
            "scalar_average": 300.0,
            "write_interval": max(write_interval * 4, 100),
        }
        tolerances = {
            "flow_rate": 0.08,
            "pressure_drop": 0.35,
            # Absolute band (anchor tolerances >= 1 are absolute) so the zero
            # datum target stays checkable.
            "outlet_pressure": 1.0,
            "source_strength": float(expected.get("tolerance") or 0.05),
            "source_response": 0.50,
            "mesh_cells": 0.0,
            "mesh_grading": 0.05,
            "force_Cd": 0.75,
            "force_Cl": 1.0,
            "scalar_average": 0.20,
            "write_interval": 0.0,
        }
        if expected.get("average_window_start"):
            # Transient time-averaged anchors. Formula values are rough and
            # carry wide absolute bands (tolerances >= 1 are absolute); the
            # in-container precompute replaces them with measured means.
            oracle.update({
                "mean_force_Cd": round(force_cd, 6),
                "mean_force_Cl_mag": 0.05,
                "mean_probe_p_mag": 0.25,
                "mean_probe_U_mag": round(velocity * 0.6, 6),
            })
            baseline.update({
                "mean_force_Cd": round(force_cd * 1.5, 6),
                "mean_force_Cl_mag": 0.5,
                "mean_probe_p_mag": 1.0,
                "mean_probe_U_mag": round(velocity * 0.2, 6),
            })
            tolerances.update({
                "mean_force_Cd": 5.0,
                "mean_force_Cl_mag": 2.0,
                "mean_probe_p_mag": 2.0,
                "mean_probe_U_mag": 2.0,
            })
        if expected.get("scalar_field"):
            # Transported-scalar anchors. Formula values are coarse with wide
            # absolute bands (tolerances >= 1 are absolute); the in-container
            # precompute replaces them with measured transported outcomes per
            # hidden variant.
            ambient = float(expected.get("scalar_ambient_value") or 300.0)
            hot = float(expected.get("scalar_hot_value") or 310.0)
            oracle.update({
                "scalar_min": round(ambient, 6),
                "scalar_max": round(hot, 6),
                "wall_scalar": round(hot, 6),
                "scalar_outlet_mean": round(ambient + 1.0, 6),
                "scalar_wall_heat_flux": 0.1,
            })
            baseline.update({
                "scalar_min": round(ambient, 6),
                "scalar_max": round(ambient, 6),
                "wall_scalar": round(ambient, 6),
                "scalar_outlet_mean": round(ambient, 6),
                "scalar_wall_heat_flux": 0.0,
            })
            tolerances.update({
                "scalar_min": 2.0,
                "scalar_max": 2.0,
                "wall_scalar": 0.05,
                "scalar_outlet_mean": 5.0,
                "scalar_wall_heat_flux": 2.0,
            })
        turbulence_model = expected.get("turbulence_model")
        if turbulence_model:
            second = "epsilon" if "epsilon" in str(turbulence_model).lower() else "omega"
            k_target = float(expected.get("turbulence_inlet_k") or 0.0)
            second_target = float(expected.get(f"turbulence_inlet_{second}") or 0.0)
            # Inlet values are dictionary-exact (Reynolds-derived); the
            # corrupted starter inflates k x100 and collapses the dissipation
            # field /100, so the baseline mirrors that defect.
            oracle["turbulence_inlet_k"] = round(k_target, 8)
            oracle[f"turbulence_inlet_{second}"] = round(second_target, 8)
            baseline["turbulence_inlet_k"] = round(k_target * 100, 8)
            baseline[f"turbulence_inlet_{second}"] = round(second_target / 100, 8)
            tolerances["turbulence_inlet_k"] = 0.10
            tolerances[f"turbulence_inlet_{second}"] = 0.10
        if candidate.family == "snappy_hex_repair":
            # The 3D obstacle force scales are not predictable by the 2D
            # heuristic above; widen to absolute bands (tolerances >= 1 are
            # absolute) until the in-container precompute measures them.
            tolerances["force_Cd"] = 5.0
            tolerances["force_Cl"] = 5.0
        anchor = {
            "changes": dict(changes),
            "oracle": oracle,
            "baseline": baseline,
            "tolerances": tolerances,
        }
        if self._can_precompute_real():
            real = self._real_anchor_metrics(candidate, variant)
            if real:
                anchor["oracle"].update(real.get("oracle", {}))
                anchor["baseline"].update(real.get("baseline", {}))
        return anchor

    def _real_anchor_metrics(self, candidate: TaskCandidate, variant: HiddenVariant) -> dict[str, dict[str, Any]] | None:
        # Optional slow path for developer/CI environments with OpenFOAM installed.
        # Formula anchors remain the normal deterministic fallback.
        try:
            with tempfile.TemporaryDirectory(prefix="synthetic-openfoam-anchor-") as tmp:
                helper = Path(tmp) / "openfoam_grader_lib.py"
                helper.write_text(GRADER_LIB)
                script = Path(tmp) / "precompute.py"
                script.write_text(
                    """
import importlib.util, json, shutil, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('openfoam_grader_lib', sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
case = Path(sys.argv[2])
expected = json.loads(sys.argv[3])
variant = json.loads(sys.argv[4])
mod.apply_hidden_variant(case, variant)
run = mod.RealOpenFoamRunner().run(case, expected, variant)
print(json.dumps(run.metrics))
"""
                )
                oracle_case = Path(tmp) / "oracle"
                baseline_case = Path(tmp) / "baseline"
                shutil.copytree(candidate.oracle_case_dir, oracle_case)
                shutil.copytree(candidate.starter_case_dir, baseline_case)
                if candidate.task_spec is not None:
                    for step in candidate.task_spec.oracle_repair.steps:
                        target = oracle_case / step.path
                        if not apply_structured_edit(target, step.old, step.new, mode=step.mode):
                            replace_required(target, step.old, step.new)
                expected = asdict(candidate.expected_metrics)
                expected["family"] = candidate.family
                variant_json = json.dumps({"name": variant.name, "changes": variant.changes})
                oracle = subprocess.run(
                    ["python3", str(script), str(helper), str(oracle_case), json.dumps(expected), variant_json],
                    text=True,
                    capture_output=True,
                    timeout=240,
                    check=False,
                )
                baseline = subprocess.run(
                    ["python3", str(script), str(helper), str(baseline_case), json.dumps(expected), variant_json],
                    text=True,
                    capture_output=True,
                    timeout=240,
                    check=False,
                )
                if oracle.returncode != 0 or baseline.returncode != 0:
                    return None
                return {"oracle": json.loads(oracle.stdout), "baseline": json.loads(baseline.stdout)}
        except Exception:
            return None

    def _pressure_drop_anchor(self, velocity: float, viscosity: float, case_kind: str) -> float:
        geometry_factor = {
            "backward_facing_step": 1.35,
            "diffuser_channel_expansion": 1.2,
            "obstacle_wake": 1.5,
            "pipe_bend": 1.4,
            "porous_pressure_drop_channel": 1.8,
        }.get(case_kind, 1.0)
        reynolds_scale = max(0.5, min(2.0, 0.01 / max(viscosity, 1e-9)))
        return max(0.05, min(5.0, 0.5 * velocity * velocity * geometry_factor * reynolds_scale))
