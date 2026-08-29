import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from utils.image import imread


def find_datasets():
    return sorted(
        dataset.name
        for dataset in Path('datasets').iterdir()
        if (dataset / 'mask').is_dir()
    )

def get_fold(split_file_path, fold_id):
    with open(split_file_path, encoding='utf-8') as file:
        folds = json.load(file)

    train_dataset_patients = defaultdict(list)
    valid_dataset_patients = defaultdict(list)
    for fold, dataset_patients in folds.items():
        patient_list = valid_dataset_patients if str(fold_id) == fold else train_dataset_patients
        for dataset, patients in dataset_patients.items():
            patient_list[dataset].extend(patients)

    train_dataset_patients = {
        dataset: sorted(patients)
        for dataset, patients in train_dataset_patients.items()
    }
    valid_dataset_patients = {
        dataset: sorted(patients)
        for dataset, patients in valid_dataset_patients.items()
    }

    return train_dataset_patients, valid_dataset_patients

class CBCTDataset(Dataset):
    def __init__(self, dataset_dir, patients, return_mask=True):
        self.dataset_dir = dataset_dir
        self.return_mask = return_mask

        self.filenames = []
        for patient in patients:
            patient_dir = self.dataset_dir / ('mask' if return_mask else 'image') / patient
            file_paths = patient_dir.iterdir()
            file_paths = sorted(file_paths, key=lambda x: int(x.stem))
            self.filenames.extend([Path(patient) / file_path.name for file_path in file_paths])

        self.length = len(self.filenames)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_path = self.dataset_dir / 'image' / filename
        image = CBCTDataset.load_image(image_path)

        if not self.return_mask:
            return image

        mask_path = self.dataset_dir / 'mask' / filename
        mask = CBCTDataset._load_mask(mask_path)

        return image, mask

    @staticmethod
    def load_image(image_path):
        image = imread(image_path)
        image = torch.from_numpy(image)
        image = image / 255
        image = image.unsqueeze(0)

        return image

    @staticmethod
    def _load_mask(mask_path):
        mask = imread(mask_path)
        mask = torch.from_numpy(mask)
        mask = mask.long()

        return mask

def get_loader(config):
    train_dataset_patients, valid_dataset_patients = get_fold(config.split_file_path, config.fold)

    train_datasets = []
    valid_datasets = []
    for dataset in find_datasets():
        train_patients = train_dataset_patients[dataset]
        valid_patients = valid_dataset_patients[dataset]

        dataset_dir = Path('datasets') / dataset

        train_dataset = CBCTDataset(dataset_dir, train_patients)
        valid_dataset = CBCTDataset(dataset_dir, valid_patients)

        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)

    train_dataset = ConcatDataset(train_datasets)
    valid_dataset = ConcatDataset(valid_datasets)

    loader_parameters = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_parameters)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_parameters)

    return train_loader, valid_loader

if __name__ == '__main__':
    from utils.config import load_config
    from utils.console import Table

    config = load_config(Path('configs') / 'segmentation.toml', fold=1)

    train_loader, valid_loader = get_loader(config)

    for images, masks in train_loader:
        break

    print('  Train Dataset')
    Table(
        ['Item', 'Shape', 'Range', 'Type'],
        ['image', images.shape, f'{images.min():.4f} ~ {images.max():.4f}', images.dtype],
        ['mask', masks.shape, f'{masks.min()} ~ {masks.max()}', masks.dtype]
    ).display()

    for images, masks in valid_loader:
        break

    print('  Valid Dataset')
    Table(
        ['Item', 'Shape', 'Range', 'Type'],
        ['image', images.shape, f'{images.min():.4f} ~ {images.max():.4f}', images.dtype],
        ['mask', masks.shape, f'{masks.min()} ~ {masks.max()}', masks.dtype]
    ).display()

