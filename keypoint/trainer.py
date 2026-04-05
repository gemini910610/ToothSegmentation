import os
import torch

from keypoint.datasets import get_loader
from keypoint.models import KeypointModel
from keypoint.metrics import EuclideanDistance
from collections import defaultdict
from src.console import track, Table
from torch import amp
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self):
        self.log_dir = os.path.join('logs', 'keypoint_flip')
        self.train_loader, self.valid_loader = get_loader('datasets/slices')
        self.model = KeypointModel().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = torch.nn.SmoothL1Loss().cuda()
        self.metric_fn = EuclideanDistance().cuda()

        self.grad_scaler = amp.GradScaler('cuda')
        self.summary_writer = SummaryWriter(self.log_dir)
        self.columns = ['Loss', 'Metric']
    def _run_epoch(self, train=True):
        if train:
            self.model.train()
            return self._run_loader(self.train_loader, 'Train', train=True)
        else:
            self.model.eval()
            return self._run_loader(self.valid_loader, 'Valid', train=False)
    def _run_loader(self, loader, desc, train=True):
        logs = defaultdict(float)

        grad_fn = torch.enable_grad if train else torch.inference_mode
        with grad_fn():
            for images, points, origin_sizes in track(loader, desc=desc):
                images = images.cuda()
                points = points.cuda()
                origin_sizes = origin_sizes.cuda()
                predicts, loss = self._model_step(images, points)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                batch_size = images.size(0)
                logs['Loss'] += loss.item() * batch_size
                self.metric_fn.update(predicts, points, origin_sizes)

        data_size = len(loader.dataset)
        metric = self.metric_fn.compute_reset()
        logs = {
            'Loss': logs['Loss'] / data_size,
            'Metric': metric
        }

        return logs
    def _model_step(self, images, points):
        with torch.autocast('cuda'):
            predicts = self.model(images)
            loss = self.criterion(predicts, points)
        return predicts, loss
    def _write_summary(self, train_data, valid_data, epoch):
        for title in self.columns:
            self.summary_writer.add_scalars(title, {'Train': train_data[title], 'Valid': valid_data[title]}, epoch)
    def fit(self, epochs):
        width = len(str(epochs))
        best_metric = torch.inf
        for epoch in range(1, epochs + 1):
            print(f'{epoch:>{width}}/{epochs}')

            train_data = self._run_epoch(train=True)
            valid_data = self._run_epoch(train=False)

            Table(
                ['', *self.columns],
                ['Train', *[train_data[column] for column in self.columns]],
                ['Valid', *[valid_data[column] for column in self.columns]]
            ).display()

            valid_metric = valid_data['Metric']
            if valid_metric <= best_metric:
                print(f'      Update Metric {best_metric:.6f} -> {valid_metric:.6f}')
                best_metric = valid_metric
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best.pth'))

            self._write_summary(train_data, valid_data, epoch)

        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'last.pth'))
        self.summary_writer.close()
