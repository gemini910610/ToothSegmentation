import torch

from keypoint.datasets import get_loader
from keypoint.models import KeypointModel
from src.console import Table, track

def evaluate_loader(loader, model, desc):
    left_distances = []
    right_distances = []

    for images, points, origin_sizes in track(loader, desc=desc):
        images = images.cuda()
        points = points.cuda()
        origin_sizes = origin_sizes.cuda()

        with torch.no_grad(), torch.autocast('cuda'):
            predicts = model(images)

        points = points.view(-1, 2, 2)
        predicts = predicts.view(-1, 2, 2)
        origin_sizes = origin_sizes.view(-1, 1, 2)
        points = points * origin_sizes
        predicts = predicts * origin_sizes

        distances = torch.norm(predicts - points, dim=-1)
        left_distances.append(distances[:, 0])
        right_distances.append(distances[:, 1])

    left_distances = torch.cat(left_distances)
    right_distances = torch.cat(right_distances)
    total_distances = torch.cat([left_distances, right_distances])

    left_mean = left_distances.mean().item()
    left_std = left_distances.std().item()
    right_mean = right_distances.mean().item()
    right_std = right_distances.std().item()
    total_mean = total_distances.mean().item()
    total_std = total_distances.std().item()

    return {
        'left': (left_mean, left_std),
        'right': (right_mean, right_std),
        'total': (total_mean, total_std),
        'size': len(left_distances)
    }

if __name__ == '__main__':
    train_loader, valid_loader = get_loader('datasets/slices')

    model = KeypointModel().cuda()
    state_dict = torch.load('logs/keypoint_baseline/best.pth')
    model.load_state_dict(state_dict)
    model.eval()

    train_data = evaluate_loader(train_loader, model, desc='Train')
    valid_data = evaluate_loader(valid_loader, model, desc='Valid')

    Table(
        ['Set', 'Size', 'Pixel Error (Left)', 'Pixel Error (Right)', 'Pixel Error (Total)'],
        ['Train', train_data['size'], f'{train_data["left"][0]:.4f}±{train_data["left"][1]:.4f}', f'{train_data["right"][0]:.4f}±{train_data["right"][1]:.4f}', f'{train_data["total"][0]:.4f}±{train_data["total"][1]:.4f}'],
        ['Valid', valid_data['size'], f'{valid_data["left"][0]:.4f}±{valid_data["left"][1]:.4f}', f'{valid_data["right"][0]:.4f}±{valid_data["right"][1]:.4f}', f'{valid_data["total"][0]:.4f}±{valid_data["total"][1]:.4f}']
    ).display()
