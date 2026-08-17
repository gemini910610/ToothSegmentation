import torch

from keypoints.dataset import get_loader
from keypoints.metrics import get_metric
from keypoints.models import load_model
from utils.console import Table, track


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.metric_fns = self._build_metric_fns(config)
        self.total_metric_fns = self._build_metric_fns(config)
        self.num_folds = config.num_folds
        self.device = config.device

    def _build_metric_fns(self, config):
        return [
            get_metric(config), # left
            get_metric(config), # right
            get_metric(config) # total
        ]

    def _update_metric_fns(self, metric_fns, predicts, points, origin_sizes):
        left_predicts, right_predicts = predicts.view(-1, 2, 2).unbind(1)
        left_points, right_points = points.view(-1, 2, 2).unbind(1)

        metric_fns[0].update(left_predicts, left_points, origin_sizes)
        metric_fns[1].update(right_predicts, right_points, origin_sizes)
        metric_fns[2].update(predicts, points, origin_sizes)

    def _run_loader(self, model, loader, desc):
        with torch.inference_mode():
            for images, points, origin_sizes in track(loader, desc):
                images = images.to(self.device, non_blocking=True)
                points = points.to(self.device, non_blocking=True)
                origin_sizes = origin_sizes.to(self.device, non_blocking=True)

                with torch.autocast(self.device):
                    predicts = model(images)

                self._update_metric_fns(self.metric_fns, predicts, points, origin_sizes)
                self._update_metric_fns(self.total_metric_fns, predicts, points, origin_sizes)

        return [
            metric_fn.compute_reset(return_std=True)
            for metric_fn in self.metric_fns
        ]

    def evaluate(self):
        logs = {}

        for fold in range(1, self.num_folds + 1):
            self.config.fold = fold
            model = load_model(self.config)
            _, loader = get_loader(self.config)

            logs[f'Fold {fold}'] = self._run_loader(model, loader, desc=f'Fold {fold}')

        logs['Mean'] = [
            metric_fn.compute_reset(return_std=True)
            for metric_fn in self.total_metric_fns
        ]

        return logs

    def format(self, logs):
        return Table(
            ['', 'Left CEJ', 'Right CEJ', 'Total CEJ'],
            *[
                [fold, *[f'{mean:.4f}±{std:.4f}' for mean, std in log]]
                for fold, log in logs.items()
            ]
        )
