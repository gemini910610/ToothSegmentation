from keypoints.dataset import get_loader
from keypoints.losses import get_loss
from keypoints.metrics import get_metric
from keypoints.models import get_model
from keypoints.optimizers import get_optimizer
from utils.trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config, get_loader, get_model, get_optimizer, get_loss, get_metric, higher_is_better=False)

    def _forward(self, batch):
        images, targets, _ = batch
        predicts = self.model(images)
        loss = self.criterion(predicts, targets)

        return predicts, loss

    def _update_metric(self, predicts, batch):
        _, targets, origin_sizes = batch
        self.metric_fn.update(predicts, targets, origin_sizes)
