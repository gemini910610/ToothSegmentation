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
