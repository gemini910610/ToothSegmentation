from torch import nn
from torchvision.models import resnet18

class KeypointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet18()
        self.net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, 4)
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    import torch

    model = KeypointModel().cuda()
    images = torch.zeros(32, 1, 224, 224, device='cuda')

    with torch.autocast('cuda'):
        predicts = model(images)

    print(f'{tuple(images.shape)} --{model.__class__.__name__}-> {tuple(predicts.shape)}')
