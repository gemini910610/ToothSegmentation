import json
import os

from collections import defaultdict
from sklearn.model_selection import GroupKFold
from src.console import Table
from src.config import load_config

config = load_config('configs/config.toml')

patients = []
pairs = []
groups = []

for dataset in config.datasets:
    image_dir = os.path.join('datasets', dataset, 'image')

    folders = os.listdir(image_dir)
    folders.sort(key=lambda x: int(x[5:])) # data_XX

    for folder in folders:
        patients.append(folder)
        images = os.listdir(os.path.join(image_dir, folder))
        pairs.extend([(dataset, folder)] * len(images))
        groups.extend([f'{dataset}/{folder}'] * len(images))

Table(
    ['Item', 'Count'],
    ['Patient', len(patients)],
    ['Image', len(groups)]
).display()

k_fold = GroupKFold(config.num_folds)

folds = {}

table = Table(['Fold', 'Train', 'Val'])
for fold, (train_indices, val_indices) in enumerate(k_fold.split(range(len(groups)), groups=groups), start=1):
    val_patients = set(pairs[index] for index in val_indices)
    val_patients = sorted(val_patients, key=lambda x: (x[0], int(x[1][5:])))
    folds[str(fold)] = defaultdict(list)
    for dataset, patients in val_patients:
        folds[str(fold)][dataset].append(patients)
    table.add_row([f'Fold {fold}', len(train_indices), len(val_indices)])
table.display()

for fold, dataset_patients in folds.items():
    print(f'   Fold {fold}')
    Table(['Dataset', 'Patient'], *dataset_patients.items()).display()

os.makedirs('splits', exist_ok=True)
with open(os.path.join('splits', f'{config.split_filename}.json'), 'w') as file:
    json.dump(folds, file, indent=4)
