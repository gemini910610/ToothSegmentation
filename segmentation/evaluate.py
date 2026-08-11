import torch

from segmentation.dataset import get_loader
from segmentation.losses import get_loss
from segmentation.metrics import get_metric
from segmentation.models import load_model
from utils.console import track


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.criterion = get_loss(config).to(config.device)
        self.metric_fn = get_metric(config).to(config.device)
        self.num_folds = config.num_folds
        self.device = config.device
        self.loss_name = self.criterion.__class__.__name__
        self.metric_name = self.metric_fn.__class__.__name__

    def _model_step(self, model, images, masks):
        with torch.autocast(self.device):
            predicts = model(images)
            loss = self.criterion(predicts, masks)
        return predicts, loss

    def _run_loader(self, model, loader, desc):
        losses = 0

        with torch.inference_mode():
            for images, masks in track(loader, desc):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                predicts, loss = self._model_step(model, images, masks)

                batch_size = images.size(0)
                losses += loss.item() * batch_size
                self.metric_fn.update(predicts, masks)

        data_size = len(loader.dataset)
        loss = losses / data_size
        metric = self.metric_fn.compute_reset()

        return loss, metric

    def evaluate(self):
        logs = {}

        for fold in range(1, self.num_folds + 1):
            self.config.fold = fold
            model = load_model(self.config)
            _, loader = get_loader(self.config)

            loss, metric = self._run_loader(model, loader, desc=f'Fold {fold}')
            logs[f'Fold {fold}'] = (loss, metric)

        return logs

if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    from utils.config import load_config
    from utils.console import Table

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(Path('logs') / experiment_name / 'segmentation.toml')

    evaluator = Evaluator(config)
    logs = evaluator.evaluate()

    Table(
        ['', evaluator.loss_name, evaluator.metric_name],
        *[
            [fold, *log]
            for fold, log in logs.items()
        ]
    ).display()
