from synthetic_openfoam_pipeline.task_families.common import family_definition


def build(task_id: str):
    return family_definition("boundary_condition_repair", task_id)

