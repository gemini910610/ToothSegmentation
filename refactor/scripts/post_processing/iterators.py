import os

from refactor.src.console import track
from refactor.src.dataset import get_fold

def iterate_fold_patients(config):
    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                yield fold, dataset, patient

def iterate_datasets(experiment_name, datasets):
    for dataset in datasets:
        dataset_path = os.path.join('outputs', experiment_name, dataset)
        for patient in track(sorted(os.listdir(dataset_path), key=lambda x: int(x[5:])), desc=dataset):
            yield dataset, patient
