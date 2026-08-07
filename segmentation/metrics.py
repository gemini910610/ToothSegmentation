import torch
from torch.nn import Module
from torchmetrics.segmentation import MeanIoU


class mIoU(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.metric_fn = MeanIoU(num_classes, include_background=False, input_format='index')
        self.num_classes = num_classes

    def update(self, predicts, targets):
        predicts = predicts.argmax(1)
        self.metric_fn.update(predicts, targets)

    def compute_reset(self):
        metric = self.metric_fn.compute()
        self.metric_fn.reset()
        return metric

METRICS = {
    'mIoU': mIoU
}

def get_metric(config):
    return METRICS[config.metric.name](**config.metric.parameters)

if __name__ == '__main__':
    from utils.console import Table

    metric_fn = mIoU(3).cuda()

    predicts = torch.randn(16, 3, 640, 640, device='cuda')
    masks = torch.randint(0, 3, (16, 640, 640), device='cuda')

    metric_fn.update(predicts, masks)
    metric = metric_fn.compute_reset()

    Table(None, ['mIoU', metric]).display()
