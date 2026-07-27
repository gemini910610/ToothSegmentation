import json
import os

from sklearn.model_selection import GroupKFold

from src.console import Table

with open('datasets/slices/test_points.json') as file:
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

Table(
    ['Item', 'Count'],
    ['Patient', len(patients)],
    ['Image', len(groups)]
).display()

k_fold = GroupKFold(4)

folds = {}

with Table(['Fold', 'Train', 'Valid']) as table:
    for fold, (train_indices, valid_indices) in enumerate(k_fold.split(range(len(groups)), groups=groups), start=1):
        valid_patients = set(groups[index] for index in valid_indices)
        valid_patients = sorted(valid_patients, key=lambda x: (x.split('/')[0], int(x.split('/')[1][5:])))
        folds[str(fold)] = valid_patients
        table.add_row([f'Fold {fold}', len(train_indices), len(valid_indices)])

os.makedirs('splits', exist_ok=True)
with open(os.path.join('splits', 'points.json'), 'w') as file:
    json.dump(folds, file, indent=4)
