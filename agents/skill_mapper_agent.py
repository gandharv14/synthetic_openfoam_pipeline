from __future__ import annotations


class SkillMapperAgent:
    FAMILY_SKILLS = {
        "boundary_condition_repair": ["boundary-condition consistency", "mass-balance validation", "hidden Reynolds robustness"],
        "postprocessing_instrumentation": ["controlDict function objects", "probe validation", "residual monitoring"],
        "solver_convergence_repair": ["fvSchemes repair", "fvSolution tolerances", "residual validation"],
        "mesh_quality_repair": ["blockMesh repair", "checkMesh quality", "cell-count preservation"],
        "source_term_calibration": ["fvOptions calibration", "source-region validation", "field metric tolerance"],
        "relaxation_factor_repair": ["SIMPLE under-relaxation strategy", "fvSolution stability", "residual convergence diagnosis"],
        "wall_patch_type_repair": ["wall boundary-condition physics", "no-slip vs slip discrimination", "wall-shear plausibility"],
        "field_dimensions_repair": ["OpenFOAM dimension sets", "field metadata consistency", "solver fatal-error diagnosis"],
        "residual_control_repair": ["residualControl semantics", "false-convergence detection", "residual-level validation"],
        "probe_placement_repair": ["probe instrumentation", "domain-extent validation", "sampling diagnostics"],
        "turbulence_bc_repair": ["RANS wall-function treatment", "freestream turbulence estimation", "k-omega/k-epsilon boundary consistency"],
        "snappy_hex_repair": ["snappyHexMesh castellation/refinement control", "3D mesh-quality budgeting", "checkMesh diagnosis"],
    }

    def map_skills(self, family: str) -> dict[str, object]:
        skills = self.FAMILY_SKILLS.get(family, [])
        return {
            "family": family,
            "skills": skills,
            "requires_multi_file_consistency": True,
            "hidden_generalization_axes": [
                "inlet velocity",
                "viscosity",
                "write interval",
                "probe location",
                "source strength",
                "mesh grading",
            ],
        }

