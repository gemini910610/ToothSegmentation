import os
import torch
import re

from argparse import ArgumentParser
from src.config import load_config
from src.models import load_model
from torchvision.transforms import ToPILImage
from src.console import track
from src.dataset import CBCTDataset
from src.downloader import ensure_experiment_exists

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('patient', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()

    experiment_name = args.exp
    fold = args.fold
    patient_root = args.patient
    output_root = args.output

    ensure_experiment_exists(experiment_name)

    config_path = os.path.join('logs', experiment_name, 'config.toml')
    config = load_config(config_path)
    config.fold = fold
    model = load_model(config)

    patients = os.listdir(patient_root)
    patients.sort(key=lambda x: tuple(
        int(x) if x.isdigit() else x
        for x in re.findall(r'\d+|\D+', x)
    ))

    to_image = ToPILImage()

    for patient in patients:
        patient_dir = os.path.join(patient_root, patient)
        output_dir = os.path.join(output_root, patient)
        os.makedirs(output_dir, exist_ok=True)
        filenames = os.listdir(patient_dir)
        for filename in track(filenames, desc=patient):
            image_path = os.path.join(patient_dir, filename)
            image = CBCTDataset.load_image(image_path)
            image = image.unsqueeze(0)
            image = image.to(config.device)
            with torch.no_grad(), torch.autocast(config.device):
                predicts = model(image)
                if isinstance(predicts, tuple):
                    predicts = predicts[0]
            predicts = predicts.argmax(1).cpu()
            predict = predicts[0].to(torch.uint8)
            predict = to_image(predict)
            output_path = os.path.join(output_dir, filename)
            predict.save(output_path)
