import torch
import os
import numpy

from torchvision.transforms import ToPILImage
from PIL import ImageFont
from PIL.ImageDraw import Draw
from src.dataset import CBCTDataset
from src.console import track
from torch.utils.data import DataLoader
from scripts.tools.widgets import Mode

def save_image(image, output_dir, filename):
    to_image = ToPILImage()
    image = to_image(image)
    image.save(os.path.join(output_dir, filename))

def save_compare(image, predict, mask, output_dir, filename):
    to_image = ToPILImage()
    font = ImageFont.load_default(20)

    width = image.size(1)

    image = torch.cat([image, predict, mask], dim=1)
    image = to_image(image)
    draw = Draw(image)
    for i, text in enumerate([filename, 'predict', 'mask']):
        draw.text((width * i, 0), text, 'white', font)
    
    image.save(os.path.join(output_dir, filename))

def predict_patient(model, dataset, patient, output_dir, config, out_image=False):
    os.makedirs(output_dir, exist_ok=True)

    dataset_dir = os.path.join('datasets', dataset)
    dataset = CBCTDataset(dataset_dir, [patient], return_mask=False)
    loader = DataLoader(dataset, config.batch_size, shuffle=False)

    image_volume = []
    mask_volume = []
    ground_truth = []
    for images, masks, filenames in track(loader, desc=f'{patient:7}'):
        image_volume.append(images.squeeze(1))
        images = images.to(config.device)

        with torch.no_grad(), torch.autocast(config.device):
            predicts = model(images) # (B, C, H, W) or (N, B, C, H, W)
            if isinstance(predicts, tuple):
                predicts = predicts[0] # (B, C, H, W)
        predicts = predicts.argmax(1).cpu() # (B, H, W)

        mask_volume.append(predicts)
        ground_truth.append(masks)

        if out_image:
            images = images.squeeze(1).cpu()

            for filename, image, predict, mask in zip(filenames, images, predicts, masks):
                image_filename = os.path.basename(filename)
                save_image(predict.to(torch.uint8), os.path.join(output_dir, 'predict'), image_filename)
                predict = predict / 2
                mask = mask / 2
                save_compare(image, predict, mask, os.path.join(output_dir, 'compare'), image_filename)

    image_volume = torch.cat(image_volume) # (B, H, W)
    image_volume = image_volume.numpy()
    image_volume = image_volume.transpose(2, 1, 0) # (W, H, Z)
    image_volume = image_volume * 255
    image_volume = image_volume.astype(numpy.uint8)
    numpy.save(os.path.join(output_dir, f'{Mode.IMAGE}.npy'), image_volume)

    mask_volume = torch.cat(mask_volume) # (B, H, W)
    mask_volume = mask_volume.numpy()
    mask_volume = mask_volume.transpose(2, 1, 0) # (W, H, Z)
    mask_volume = mask_volume.astype(numpy.uint8)
    numpy.save(os.path.join(output_dir, f'{Mode.PREDICT}.npy'), mask_volume)

    ground_truth = torch.cat(ground_truth) # (B, H, W)
    ground_truth = ground_truth.numpy()
    ground_truth = ground_truth.transpose(2, 1, 0) # (W, H, Z)
    ground_truth = ground_truth.astype(numpy.uint8)
    numpy.save(os.path.join(output_dir, f'{Mode.GROUND_TRUTH}.npy'), ground_truth)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from src.config import load_config
    from src.dataset import get_fold
    from src.models import load_model
    from src.downloader import ensure_experiment_exists

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--out-image', action='store_true')
    args = parser.parse_args()

    experiment_name = args.exp
    out_image = args.out_image
    ensure_experiment_exists(experiment_name)

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold in range(1, config.num_folds + 1):
        config.fold = fold

        model = load_model(config)

        _, val_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in val_dataset_patients.items():
            for patient in patients:
                output_dir = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient)
                if out_image:
                    os.makedirs(os.path.join(output_dir, 'predict'), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, 'compare'), exist_ok=True)

                predict_patient(model, dataset, patient, output_dir, config, out_image)
