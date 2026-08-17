from pathlib import Path

import torch


def get_model(config):
    match config.model.name:
        case 'ResNet-18':
            from keypoints.models.resnet18 import ResNet18

            return ResNet18(**config.model.parameters).to(config.device)

def load_model(config):
    model = get_model(config)
    state_dict = torch.load(Path('logs') / config.experiment / f'Fold_{config.fold}' / 'best.pth')
    model.load_state_dict(state_dict)
    model.eval()

    return model
