import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.image import imread


def get_fold(split_file_path, fold_id):
    with open(split_file_path, encoding='utf-8') as file:
        folds = json.load(file)

    train_patients = []
    valid_patients = []
    for fold, dataset_patients in folds.items():
        patient_list = valid_patients if str(fold_id) == fold else train_patients
        for dataset, patients in dataset_patients.items():
            patient_list.extend(f'{dataset}/{patient}' for patient in patients)

    train_patients.sort()
    valid_patients.sort()

    return train_patients, valid_patients

class KeypointDataset(Dataset):
    def __init__(self, dataset_dir, patients):
        self.dataset_dir = dataset_dir

        with open(dataset_dir / 'keypoints.json', encoding='utf-8') as file:
            data = json.load(file)

        self.points = []
        for patient in patients:
            for label, degrees in data[patient].items():
                for degree, points in degrees.items():
                    folder = Path(patient) / label
                    self.points.append((folder, degree, points, True))
                    self.points.append((folder, degree, points, False))

        self.length = len(self.points)

        self.transforms = transforms.Resize((224, 224))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        folder, degree, ((left_x, left_y), (right_x, right_y)), flip = self.points[index]
        image = KeypointDataset.load_image(self.dataset_dir / folder, degree)

        _, height, width = image.shape
        origin_size = torch.tensor([width, height])
        image = self.transforms(image)

        if flip:
            image = image.flip(2)
            left_x, left_y, right_x, right_y = 1 - right_x, right_y, 1 - left_x, left_y
        points = torch.tensor([left_x, left_y, right_x, right_y])

        return image, points, origin_size

    @staticmethod
    def load_image(folder, degree):
        image = imread(folder / f'image_{degree}.png')
        segmentation = imread(folder / f'segmentation_{degree}.png')

        image = image * (segmentation == 127)
        image = torch.from_numpy(image)
        image = image / 255
        image = image.unsqueeze(0)

        return image

def get_loader(config):
    train_patients, valid_patients = get_fold(config.split_file_path, config.fold)

    dataset_dir = Path('datasets') / 'Slice'

    train_dataset = KeypointDataset(dataset_dir, train_patients)
    valid_dataset = KeypointDataset(dataset_dir, valid_patients)

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

    config = load_config(Path('configs') / 'keypoints.toml', fold=1)

    train_loader, valid_loader = get_loader(config)

    for images, points, origin_sizes in train_loader:
        break

    print('  Train Dataset')
    Table(
        ['Item', 'Shape', 'Range', 'Type'],
        ['image', images.shape, f'{images.min():.4f} ~ {images.max():.4f}', images.dtype],
        ['point', points.shape, f'{points.min():.4f} ~ {points.max():.4f}', points.dtype],
        ['origin size', origin_sizes.shape, f'{origin_sizes.min():>6} ~ {origin_sizes.max():<6}', origin_sizes.dtype]
    ).display()

    for images, points, origin_sizes in valid_loader:
        break

    print('  Valid Dataset')
    Table(
        ['Item', 'Shape', 'Range', 'Type'],
        ['image', images.shape, f'{images.min():.4f} ~ {images.max():.4f}', images.dtype],
        ['point', points.shape, f'{points.min():.4f} ~ {points.max():.4f}', points.dtype],
        ['origin size', origin_sizes.shape, f'{origin_sizes.min():>6} ~ {origin_sizes.max():<6}', origin_sizes.dtype]
    ).display()
