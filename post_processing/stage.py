from constants.mode import Mode
from visualize.utils import JsonHandler


def load_stages(patient_dir):
    return set(JsonHandler.load(patient_dir / 'stage.json'))

def has_completed(patient_dir, mode):
    return mode in load_stages(patient_dir)

def advance_stage(patient_dir, mode):
    order = Mode.items()
    index = order.index(mode)

    stages = {stage for stage in load_stages(patient_dir) if order.index(stage) <= index}
    stages.add(mode)

    JsonHandler.save(patient_dir / 'stage.json', {stage: True for stage in stages})
