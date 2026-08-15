from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from segmentation.dataset import get_loader
from segmentation.losses import get_loss
from segmentation.metrics import get_metric
from segmentation.models import get_model
from segmentation.optimizers import get_optimizer
from utils.console import Table, track


class Trainer:
    def __init__(self, config):
        self.log_dir = Path('logs') / config.experiment / f'Fold_{config.fold}'
        self.train_loader, self.valid_loader = get_loader(config)
        self.model = get_model(config).to(config.device)
        self.optimizer = get_optimizer(self.model, config)
        self.criterion = get_loss(config).to(config.device)
        self.metric_fn = get_metric(config).to(config.device)
        self.device = config.device
        self.loss_name = self.criterion.__class__.__name__
        self.metric_name = self.metric_fn.__class__.__name__

        self.grad_scaler = GradScaler(config.device)
        self.summary_writer = SummaryWriter(self.log_dir)

    def _model_step(self, images, masks):
        with torch.autocast(self.device):
            predicts = self.model(images)
            loss = self.criterion(predicts, masks)

        return predicts, loss

    def _run_loader(self, loader, desc, train=True):
        losses = 0

        grad_fn = torch.enable_grad if train else torch.inference_mode
        with grad_fn():
            for images, masks in track(loader, desc):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                predicts, loss = self._model_step(images, masks)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                batch_size = images.size(0)
                losses += loss.item() * batch_size
                self.metric_fn.update(predicts, masks)

        data_size = len(loader.dataset)
        loss = losses / data_size
        metric = self.metric_fn.compute_reset()

        return loss, metric

    def _run_epoch(self, train=True):
        if train:
            self.model.train()

            return self._run_loader(self.train_loader, 'Train', train=True)
        else:
            self.model.eval()

            return self._run_loader(self.valid_loader, 'Valid', train=False)

    def _write_summary(self, train_loss, train_metric, valid_loss, valid_metric, epoch):
        self.summary_writer.add_scalars(self.loss_name, {'Train': train_loss, 'Valid': valid_loss}, epoch)
        self.summary_writer.add_scalars(self.metric_name, {'Train': train_metric, 'Valid': valid_metric}, epoch)

    def fit(self, epochs):
        width = len(str(epochs))
        best_metric = 0
        for epoch in range(1, epochs + 1):
            print(f'{epoch:>{width}}/{epochs}')

            train_loss, train_metric = self._run_epoch(train=True)
            valid_loss, valid_metric = self._run_epoch(train=False)

            Table(
                ['', self.loss_name, self.metric_name],
                ['Train', train_loss, train_metric],
                ['Valid', valid_loss, valid_metric]
            ).display()

            if valid_metric >= best_metric:
                print(f'      Update {self.metric_name} {best_metric:.6f} -> {valid_metric:.6f}')
                best_metric = valid_metric
                torch.save(self.model.state_dict(), self.log_dir / 'best.pth')

            self._write_summary(train_loss, train_metric, valid_loss, valid_metric, epoch)

        self.summary_writer.close()
