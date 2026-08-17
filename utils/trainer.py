from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from utils.console import Table, track


class BaseTrainer:
    def __init__(self, config, get_loader, get_model, get_optimizer, get_loss, get_metric, higher_is_better=True):
        self.log_dir = Path('logs') / config.experiment / f'Fold_{config.fold}'
        self.train_loader, self.valid_loader = get_loader(config)
        self.model = get_model(config)
        self.optimizer = get_optimizer(self.model, config)
        self.criterion = get_loss(config)
        self.metric_fn = get_metric(config)

        self.higher_is_better = higher_is_better
        self.device = config.device
        self.loss_name = self.criterion.__class__.__name__
        self.metric_name = self.metric_fn.__class__.__name__

        self.grad_scaler = GradScaler(config.device)
        self.summary_writer = SummaryWriter(self.log_dir)

    def _forward(self, batch):
        raise NotImplementedError

    def _update_metric(self, predicts, batch):
        raise NotImplementedError

    def _run_loader(self, loader, desc, train=True):
        losses = 0

        grad_fn = torch.enable_grad if train else torch.inference_mode
        with grad_fn():
            for batch in track(loader, desc):
                batch = tuple(item.to(self.device, non_blocking=True) for item in batch)

                with torch.autocast(self.device):
                    predicts, loss = self._forward(batch)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                batch_size = batch[0].size(0)
                losses += loss.item() * batch_size
                self._update_metric(predicts, batch)

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

    def _is_better(self, metric, best_metric):
        return metric >= best_metric if self.higher_is_better else metric <= best_metric

    def fit(self, epochs):
        width = len(str(epochs))
        best_metric = -torch.inf if self.higher_is_better else torch.inf

        for epoch in range(1, epochs + 1):
            print(f'{epoch:>{width}}/{epochs}')

            train_loss, train_metric = self._run_epoch(train=True)
            valid_loss, valid_metric = self._run_epoch(train=False)

            Table(
                ['', self.loss_name, self.metric_name],
                ['Train', train_loss, train_metric],
                ['Valid', valid_loss, valid_metric]
            ).display()

            if self._is_better(valid_metric, best_metric):
                print(f'      Update {self.metric_name} {best_metric:.6f} -> {valid_metric:.6f}')
                best_metric = valid_metric
                torch.save(self.model.state_dict(), self.log_dir / 'best.pth')

            self._write_summary(train_loss, train_metric, valid_loss, valid_metric, epoch)

        self.summary_writer.close()
