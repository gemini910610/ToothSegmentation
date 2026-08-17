from torch import nn

LOSSES = {
    'SmoothL1Loss': nn.SmoothL1Loss
}

def get_loss(config):
    return LOSSES[config.loss.name](**config.loss.parameters).to(config.device)

if __name__ == '__main__':
    import torch

    from utils.console import Table

    criterion = nn.SmoothL1Loss().cuda()

    predicts = torch.randn(32, 4, device='cuda')
    targets = torch.randn(32, 4, device='cuda')

    with torch.autocast('cuda'):
        loss = criterion(predicts, targets)

    Table(None, ['Loss', loss]).display()
