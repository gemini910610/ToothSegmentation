import json
from pathlib import Path


def group_data(config):
    with open(Path('datasets') / config.annotation_file, encoding='utf-8') as file:
        data = json.load(file)

    patients = []
    groups = []

    for patient, patient_data in data.items():
        patients.append(patient)
        count = sum(
            len(label_data)
            for label_data in patient_data.values()
        )
        groups.extend([patient] * count)

    return patients, groups
