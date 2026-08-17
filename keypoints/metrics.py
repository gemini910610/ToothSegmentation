import torch
from torchmetrics import Metric


class PixelError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, predicts, targets, origin_sizes):
        predicts = predicts.view(-1, 2, 2)
        targets = targets.view(-1, 2, 2)
        origin_sizes = origin_sizes.view(-1, 1, 2)

        predicts = predicts * origin_sizes
        targets = targets * origin_sizes

        distances = torch.norm(predicts - targets, dim=-1)
        self.total += distances.sum()
        self.count += distances.numel()

    def compute(self):
        return self.total / self.count

    def compute_reset(self):
        metric = self.compute()
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
