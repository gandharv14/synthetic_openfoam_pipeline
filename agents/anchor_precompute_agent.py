from __future__ import annotations

from dataclasses import asdict
from typing import Any

from synthetic_openfoam_pipeline.schemas import HiddenVariant, TaskCandidate


class AnchorPrecomputeAgent:
    def attach_anchors(self, candidate: TaskCandidate) -> TaskCandidate:
        anchors = {
            "version": 1,
            "family": candidate.family,
            "seed_case": candidate.seed_case.seed_id,
            "mode": "deterministic-formula",
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

    def _variant_anchor(self, candidate: TaskCandidate, variant: HiddenVariant) -> dict[str, Any]:
        changes = variant.changes
        expected = asdict(candidate.expected_metrics)
        velocity = float(changes.get("inlet_velocity", candidate.seed_case.parameters.get("inlet_velocity", 1.0)))
        viscosity = float(changes.get("viscosity", candidate.seed_case.parameters.get("viscosity", 0.01)))
        source = float(changes.get("source_strength", expected.get("source_target") or candidate.seed_case.parameters.get("source_strength", 0.2)))
        grading = float(changes.get("mesh_grading", 1.5))
        write_interval = int(changes.get("write_interval", 20))
        area = self._inlet_area(candidate.seed_case.case_kind)
        flow_rate = velocity * area
        pressure_drop = self._pressure_drop_anchor(velocity, viscosity, candidate.seed_case.case_kind)
        mesh_cells = self._mesh_cells(candidate.seed_case.case_kind)
        scalar_average = 305.0 if "cavity" in candidate.seed_case.case_kind else 300.0
        force_cd = 0.35 + 0.05 * velocity
        force_cl = 0.02

        oracle = {
            "flow_rate": round(flow_rate, 6),
            "pressure_drop": round(pressure_drop, 6),
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
            "source_strength": float(expected.get("tolerance") or 0.05),
            "source_response": 0.50,
            "mesh_cells": 0.0,
            "mesh_grading": 0.05,
            "force_Cd": 0.75,
            "force_Cl": 1.0,
            "scalar_average": 0.20,
            "write_interval": 0.0,
        }
        return {
            "changes": dict(changes),
            "oracle": oracle,
            "baseline": baseline,
            "tolerances": tolerances,
        }

    def _inlet_area(self, case_kind: str) -> float:
        if case_kind in {"cavity", "heated_cavity"}:
            return 0.1
        if case_kind == "backward_facing_step":
            return 0.06
        if case_kind == "diffuser_channel_expansion":
            return 0.06
        if case_kind == "pipe_bend":
            return 0.04
        return 0.1

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

    def _mesh_cells(self, case_kind: str) -> int:
        return {
            "cavity": 100,
            "heated_cavity": 100,
            "backward_facing_step": 224,
            "diffuser_channel_expansion": 224,
            "obstacle_wake": 262,
            "pipe_bend": 125,
        }.get(case_kind, 200)
