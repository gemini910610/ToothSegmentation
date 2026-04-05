import torch

from torchmetrics import Metric

class EuclideanDistance(Metric):
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
