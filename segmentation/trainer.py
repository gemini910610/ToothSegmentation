from segmentation.dataset import get_loader
from segmentation.losses import get_loss
from segmentation.metrics import get_metric
from segmentation.models import get_model
from segmentation.optimizers import get_optimizer
from utils.trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config, get_loader, get_model, get_optimizer, get_loss, get_metric, higher_is_better=True)

    def _forward(self, batch):
        images, masks = batch
        predicts = self.model(images)
        loss = self.criterion(predicts, masks)

        return predicts, loss

    def _update_metric(self, predicts, batch):
        _, masks = batch
        self.metric_fn.update(predicts, masks)
