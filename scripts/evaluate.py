import os
import torch

from src.models import get_model

def evaluate_fold(config, criterion, metric_fn):
    model = load_model(config)
    _, loader = get_loader(config)

    logs = defaultdict(float)
    data_size = 0

    for images, masks, _ in track(loader, desc=f'Fold {config.FOLD}'):
        images = images.to(config.DEVICE)
        masks = masks.to(config.DEVICE)

        zero_mask = (masks == 0).all((1, 2))
        images = images[~zero_mask]
        masks = masks[~zero_mask]

        batch_size = images.size(0)
        data_size += batch_size
        if batch_size == 0:
            continue

        with torch.no_grad(), torch.autocast(config.DEVICE):
            predicts = model(images)
            loss_dict = criterion(predicts, masks)

        for name, loss in loss_dict.items():
            logs[name] += loss.item() * batch_size
        metric_fn.update(predicts, masks)

    logs = {
        name: loss / data_size
        for name, loss in logs.items()
    }
    logs[config.METRIC_NAME] = metric_fn.compute_reset().item()

    return logs

def load_model(config):
    model = get_model(config).to(config.DEVICE)
    state_dict = torch.load(os.path.join('logs', config.EXPERIMENT, f'Fold_{config.FOLD}', 'best.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

if __name__ == '__main__':
    from argparse import ArgumentParser
    from src.utils import load_config, track, Table
    from src.metrics import get_metric
    from src.losses import get_loss
    from src.dataset import get_loader
    from collections import defaultdict

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.json'))

    metric_fn = get_metric(config).to(config.DEVICE)
    criterion = get_loss(config).to(config.DEVICE)

    logs = {}
    for fold in range(1, config.NUM_FOLDS + 1):
        config.FOLD = fold
        logs[f'Fold {fold}'] = evaluate_fold(config, criterion, metric_fn)

    titles = [*criterion.components, config.METRIC_NAME]
    Table(
        ['', *titles],
        *[
            [fold, *[log[title] for title in titles]]
            for fold, log in logs.items()
        ]
    ).display()
