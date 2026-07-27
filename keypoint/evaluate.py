import torch

from keypoint.datasets import get_loader
from keypoint.models import KeypointModel
from src.console import track

def evaluate_fold(experiment_name, fold):
    model = KeypointModel().cuda()
    state_dict = torch.load(f'logs/{experiment_name}/Fold_{fold}/best.pth')
    model.load_state_dict(state_dict)
    model.eval()

    _, loader = get_loader('datasets/slices', fold)

    logs = evaluate_loader(loader, model, desc=f'Fold {fold}')

    return {
        'Left CEJ': f'{logs["left"][0]:.4f}±{logs["left"][1]:.4f}',
        'Right CEJ': f'{logs["right"][0]:.4f}±{logs["right"][1]:.4f}',
        'Total CEJ': f'{logs["total"][0]:.4f}±{logs["total"][1]:.4f}',
        'left_distances': logs['left_distances'],
        'right_distances': logs['right_distances']
    }

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
        'size': len(left_distances),
        'left_distances': left_distances,
        'right_distances': right_distances
    }

if __name__ == '__main__':
    from argparse import ArgumentParser

    from src.console import Table

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    logs = {
        f'Fold {fold}': evaluate_fold(experiment_name, fold)
        for fold in range(1, 5)
    }

    titles = ['Left CEJ', 'Right CEJ', 'Total CEJ']
    with Table(['', *titles]) as table:
        left_distances = []
        right_distances = []

        for fold, log in logs.items():
            table.add_row([fold, *[log[title] for title in titles]])
            left_distances.append(log['left_distances'])
            right_distances.append(log['right_distances'])

        left_distances = torch.cat(left_distances)
        right_distances = torch.cat(right_distances)
        total_distances = torch.cat([left_distances, right_distances])

        left_mean = left_distances.mean().item()
        left_std = left_distances.std().item()
        right_mean = right_distances.mean().item()
        right_std = right_distances.std().item()
        total_mean = total_distances.mean().item()
        total_std = total_distances.std().item()

        table.add_row(['Mean', f'{left_mean:.4f}±{left_std:.4f}', f'{right_mean:.4f}±{right_std:.4f}', f'{total_mean:.4f}±{total_std:.4f}'])
