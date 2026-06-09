from synthetic_openfoam_pipeline.task_families.common import family_definition


def build(task_id: str):
    return family_definition("solver_convergence_repair", task_id)

