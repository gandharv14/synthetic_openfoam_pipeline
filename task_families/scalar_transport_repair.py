from synthetic_openfoam_pipeline.task_families.common import family_definition


def build(task_id: str, seed_selector=None):
    return family_definition("scalar_transport_repair", task_id, seed_selector)
