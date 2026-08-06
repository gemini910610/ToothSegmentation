import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import GroupKFold

from utils.config import load_config
from utils.console import Table

parser = ArgumentParser()
parser.add_argument('task', type=str, choices=['segmentation', 'keypoints'])
args = parser.parse_args()

task = args.task

match task:
    case 'segmentation':
        from segmentation.kfold import group_data
    case 'keypoints':
        from keypoints.kfold import group_data

config = load_config(Path('configs') / f'{task}.toml')

patients, groups = group_data(config)

Table(
    ['Item', 'Count'],
    ['Patient', len(patients)],
    ['Image', len(groups)]
).display()

kfold = GroupKFold(config.num_folds)
splits = kfold.split(range(len(groups)), groups=groups)

folds = {}

with Table(['Fold', 'Train', 'Valid']) as table:
    for fold, (train_indices, valid_indices) in enumerate(splits, 1):
        valid_patients = {tuple(groups[index].split('/')) for index in valid_indices}
        valid_patients = sorted(valid_patients)
        folds[str(fold)] = defaultdict(list)
        for dataset, patient in valid_patients:
            folds[str(fold)][dataset].append(patient)
        table.add_row([f'Fold {fold}', len(train_indices), len(valid_indices)])

for fold, dataset_patients in folds.items():
    print(f'  Fold {fold}')
    Table(
        ['Dataset', 'Patient'],
        *[
            (dataset, ', '.join(patients))
            for dataset, patients in dataset_patients.items()
        ]
    ).display()

output_dir = Path('splits')
output_dir.mkdir(exist_ok=True)

with open(output_dir / f'{config.split_filename}.json', 'w', encoding='utf-8') as file:
    json.dump(folds, file, indent=4)
