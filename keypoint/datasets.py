import cv2
import json
import os
import torch

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class KeypointDataset(Dataset):
    def __init__(self, dataset_dir, patients):
        self.dataset_dir = dataset_dir

        with open(os.path.join(dataset_dir, 'points_refine.json')) as file:
            data = json.load(file)

        self.points = []
        for patient in patients:
            for label, degrees in data[patient].items():
                for degree, points in degrees.items():
                    filename = os.path.join(patient, label, f'tooth_{degree}.png')
                    self.points.append((filename, points, True))
                    self.points.append((filename, points, False))

        self.length = len(self.points)

        self.transforms = transforms.Resize((224, 224))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        filename, ((left_x, left_y), (right_x, right_y)), flip = self.points[index]
        image_path = os.path.join(self.dataset_dir, filename)
        image = KeypointDataset.load_image(image_path)
        _, height, width = image.shape
        origin_size = torch.tensor([width, height])
        image = self.transforms(image)
        if flip:
            image = image.flip(2)
            left_x, left_y, right_x, right_y = 1 - right_x, right_y, 1 - left_x, left_y
        points = torch.tensor([left_x, left_y, right_x, right_y])
        return image, points, origin_size

    def load_image(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(image)
        image = image / 255
        image = image.unsqueeze(0)
        return image

def get_loader(dataset_dir):
    with open(f'{dataset_dir}/points_refine.json') as file:
        data = json.load(file)

    groups = []
    for patient, labels in data.items():
        groups.extend([patient] * len(labels))
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    for train_indices, valid_indices in splitter.split(groups, groups=groups):
        break

    train_patients = sorted({groups[index] for index in train_indices}, key=lambda x: (x.split('/')[0], int(x.split('/')[1][5:])))
    valid_patients = sorted({groups[index] for index in valid_indices}, key=lambda x: (x.split('/')[0], int(x.split('/')[1][5:])))

    train_dataset = KeypointDataset(dataset_dir, train_patients)
    valid_dataset = KeypointDataset(dataset_dir, valid_patients)

    loader_parameters = {
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_parameters)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_parameters)
    return train_loader, valid_loader

if __name__ == '__main__':
    from src.console import Table

    train_loader, valid_loader = get_loader('datasets/slices')

    for images, points, origin_sizes in train_loader:
        break

    Table(
        ['Item', 'Shape', 'Range', 'Type'],
        ['image', images.shape, f'{images.min():.4f} ~ {images.max():.4f}', images.dtype],
        ['points', points.shape, f'{points.min():.4f} ~ {points.max():.4f}', points.dtype],
        ['origin_size', origin_sizes.shape, f'{origin_sizes.min():>6} ~ {origin_sizes.max():<6}', origin_sizes.dtype]
    ).display()
