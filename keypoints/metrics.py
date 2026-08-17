import torch
from torchmetrics import Metric


class PixelError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('distances', default=[], dist_reduce_fx='cat')

    def update(self, predicts, targets, origin_sizes):
        batch_size = origin_sizes.size(0)
        predicts = predicts.view(batch_size, -1, 2)
        targets = targets.view(batch_size, -1, 2)
        origin_sizes = origin_sizes.view(batch_size, 1, 2)

        predicts = predicts * origin_sizes
        targets = targets * origin_sizes

        distances = torch.norm(predicts - targets, dim=-1)
        self.distances.append(distances.flatten())

    def compute(self, return_std=False):
        distances = torch.cat(self.distances)
        mean = distances.mean().item()

        if not return_std:
            return mean

        std = distances.std().item()

        return mean, std

    def compute_reset(self, return_std=False):
        metric = self.compute(return_std)
        self.reset()

        return metric

METRICS = {
    'PixelError': PixelError
}

def get_metric(config):
    return METRICS[config.metric.name](**config.metric.parameters).to(config.device)

if __name__ == '__main__':
    from utils.console import Table

    metric_fn = PixelError().cuda()

    predicts = torch.randn(32, 4, device='cuda')
    targets = torch.randn(32, 4, device='cuda')
    origin_sizes = torch.randint(100, 200, (32, 2), device='cuda')

    metric_fn.update(predicts, targets, origin_sizes)
    metric = metric_fn.compute_reset()

    Table(None, ['PixelError', metric]).display()
