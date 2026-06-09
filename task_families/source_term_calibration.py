from synthetic_openfoam_pipeline.task_families.common import family_definition


def build(task_id: str):
    return family_definition("source_term_calibration", task_id)

