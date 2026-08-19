from pathlib import Path

import numpy
from torch.utils.data import DataLoader

from segmentation.dataset import CBCTDataset
from segmentation.predict import predict_volumes


def predict_patient(model, dataset, patient, config):
    dataset = CBCTDataset(Path('datasets') / dataset, [patient], return_mask=False)
    loader = DataLoader(dataset, config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return predict_volumes(model, loader, config, desc=patient)

if __name__ == '__main__':
    from argparse import ArgumentParser

    from constants.mode import Mode
    from segmentation.dataset import get_fold
    from segmentation.models import load_model
    from utils.config import load_config

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('best_fold', type=int)
    parser.add_argument('datasets', nargs='+', type=str)
    args = parser.parse_args()

    experiment_name = args.exp
    best_fold = args.best_fold
    datasets = args.datasets

    config = load_config(Path('logs') / experiment_name / 'segmentation.toml')

    processed_patients = set()

    for fold in range(1, config.num_folds + 1):
        config.fold = fold

        model = load_model(config)

        _, dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in dataset_patients.items():
            if dataset not in datasets:
                continue

            print(f'Fold {fold} {dataset}')
            for patient in patients:
                image_volume, predict_volume = predict_patient(model, dataset, patient, config)
                output_dir = Path('outputs') / experiment_name / dataset / patient
                output_dir.mkdir(parents=True, exist_ok=True)
                numpy.save(output_dir / f'{Mode.IMAGE}.npy', image_volume)
                numpy.save(output_dir / f'{Mode.PREDICT}.npy', predict_volume)
                processed_patients.add(f'{dataset}/{patient}')

    config.fold = best_fold
    model = load_model(config)

    for dataset in datasets:
        print(dataset)
        for patient in (Path('datasets') / dataset / 'image').iterdir():
            patient = patient.name
            if f'{dataset}/{patient}' in processed_patients:
                continue

            image_volume, predict_volume = predict_patient(model, dataset, patient, config)
            output_dir = Path('outputs') / experiment_name / dataset / patient
            output_dir.mkdir(parents=True, exist_ok=True)
            numpy.save(output_dir / f'{Mode.IMAGE}.npy', image_volume)
            numpy.save(output_dir / f'{Mode.PREDICT}.npy', predict_volume)
