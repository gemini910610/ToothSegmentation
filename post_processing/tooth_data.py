import numpy

from visualize.utils import JsonHandler


def load_tooth_data(patient_dir):
    data = JsonHandler.load(patient_dir / 'data.json')

    return {int(label): value for label, value in data.items()}

def save_tooth_data(patient_dir, data):
    JsonHandler.save(patient_dir / 'data.json', {str(label): value for label, value in data.items()})

def set_entry(data, label, value):
    if value in (None, 'Normal Tooth'):
        data.pop(label, None)
    else:
        data[label] = value

def prune_stale_ids(data, valid_ids):
    for label in list(data):
        if label not in valid_ids:
            del data[label]

def is_excluded(value):
    return not value.isdigit()

def resolve_volume(patient_dir, tooth_volume):
    tooth_data = load_tooth_data(patient_dir)

    lookup_table = numpy.zeros(tooth_volume.max() + 1, dtype=numpy.uint8)
    for label, value in tooth_data.items():
        if not is_excluded(value):
            lookup_table[label] = int(value)

    return lookup_table[tooth_volume]
