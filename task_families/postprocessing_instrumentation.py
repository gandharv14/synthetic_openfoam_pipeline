from synthetic_openfoam_pipeline.task_families.common import family_definition


def build(task_id: str, seed_selector=None):
    return family_definition("postprocessing_instrumentation", task_id, seed_selector)

