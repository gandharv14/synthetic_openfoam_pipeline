from __future__ import annotations


class SkillMapperAgent:
    FAMILY_SKILLS = {
        "boundary_condition_repair": ["boundary-condition consistency", "mass-balance validation", "hidden Reynolds robustness"],
        "postprocessing_instrumentation": ["controlDict function objects", "probe validation", "residual monitoring"],
        "solver_convergence_repair": ["fvSchemes repair", "fvSolution tolerances", "residual validation"],
        "mesh_quality_repair": ["blockMesh repair", "checkMesh quality", "cell-count preservation"],
        "source_term_calibration": ["fvOptions calibration", "source-region validation", "field metric tolerance"],
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

