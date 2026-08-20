from torch import nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        self.net = resnet18()
        self.net.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_keypoints * 2)

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    import torch

    keypoints_model = ResNet18(1, 2).cuda()

    images = torch.zeros(32, 1, 224, 224, device='cuda')

    with torch.autocast('cuda'):
        predicts = keypoints_model(images)

    print(f'{tuple(images.shape)} --{keypoints_model.__class__.__name__}-> {tuple(predicts.shape)}')
