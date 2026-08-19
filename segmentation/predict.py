import numpy
import torch
from torch.utils.data import DataLoader, Dataset

from segmentation.dataset import CBCTDataset
from utils.console import track


class ImageDirDataset(Dataset):
    def __init__(self, image_dir):
        self.file_paths = sorted(image_dir.iterdir(), key=lambda path: int(path.stem))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        return CBCTDataset.load_image(self.file_paths[index])

def predict_volumes(model, loader, config, desc=''):
    image_volume = []
    predict_volume = []
    for images in track(loader, desc=desc):
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

def predict_patient(model, image_dir, config):
    dataset = ImageDirDataset(image_dir)
    loader = DataLoader(dataset, config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return predict_volumes(model, loader, config, desc=image_dir.parent.name)
