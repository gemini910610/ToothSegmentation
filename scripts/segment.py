from pathlib import Path

import numpy
import torch
from torch.utils.data import DataLoader

from segmentation.dataset import CBCTDataset
from utils.console import track


def predict_patient(model, dataset, patient, config):
    dataset = CBCTDataset(Path('datasets') / dataset, [patient], return_mask=False)
    loader = DataLoader(dataset, config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    image_volume = []
    predict_volume = []
    for images in track(loader, desc=patient):
        image_volume.append(images.squeeze(1))
        images = images.to(config.device, non_blocking=True)

        with torch.no_grad(), torch.autocast(config.device):
            predicts = model(images) # (B, C, H, W)
        predicts = predicts.argmax(1).cpu() # (B, H, W)

        predict_volume.append(predicts)

    image_volume = torch.cat(image_volume) # (Z, H, W)
    image_volume = image_volume.numpy()
    image_volume = image_volume.transpose(2, 1, 0) # (W, H, Z)
    image_volume = image_volume * 255
    image_volume = image_volume.astype(numpy.uint8)

    predict_volume = torch.cat(predict_volume) # (Z, H, W)
    predict_volume = predict_volume.numpy()
    predict_volume = predict_volume.transpose(2, 1, 0) # (W, H, Z)
    predict_volume = predict_volume.astype(numpy.uint8)

    return image_volume, predict_volume

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
