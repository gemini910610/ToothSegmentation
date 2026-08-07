from torch import nn
from torch.nn.functional import one_hot


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predicts, targets):
        predicts = predicts.softmax(1) # (B, C, H, W)
        predicts = predicts.transpose(0, 1) # (C, B, H, W)
        predicts = predicts.flatten(1) # (C, BHW)

        targets = one_hot(targets, self.num_classes) # (B, H, W, C)
        targets = targets.permute(3, 0, 1, 2) # (C, B, H, W)
        targets = targets.flatten(1) # (C, BHW)

        # ignore background
        predicts = predicts[1:]
        targets = targets[1:]

        intersection = (predicts * targets).sum(1) # (C)
        predict_area = predicts.sum(1) # (C)
        target_area = targets.sum(1) # (C)

        dice_loss = 1 - (2 * intersection + 1e-6) / (predict_area + target_area + 1e-6)
        dice_loss = dice_loss.mean()
        return dice_loss

class DiceCELoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.dice_loss = DiceLoss(num_classes)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, predicts, targets):
        dice_loss = self.dice_loss(predicts, targets)
        cross_entropy_loss = self.cross_entropy_loss(predicts, targets)
        loss = 0.5 * dice_loss + 0.5 * cross_entropy_loss
        return loss

LOSSES = {
    'DiceCELoss': DiceCELoss
}

def get_loss(config):
    return LOSSES[config.loss.name](**config.loss.parameters)

if __name__ == '__main__':
    import torch

    from utils.console import Table

    criterion = DiceCELoss(3).cuda()

    predicts = torch.randn(16, 3, 640, 640, device='cuda')
    masks = torch.randint(0, 3, (16, 640, 640), device='cuda')

    with torch.autocast('cuda'):
        loss = criterion(predicts, masks)

    Table(None, ['Loss', loss]).display()
