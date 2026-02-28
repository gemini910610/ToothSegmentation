import torch

from torch import nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(in_channels, 64),
            Block(64, 128),
            Block(128, 256),
            Block(256, 512)
        ])
        self.downsample = nn.MaxPool2d(2)
    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.downsample(x)
        return x, features

class PositionEncoder(nn.Module):
    def __init__(self, height, width, dim):
        super().__init__()
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing='ij')
        quarter = dim // 4
        omega = torch.arange(quarter, dtype=torch.float32) / quarter
        omega = 1 / (10000 ** omega)
        omega = omega.reshape(1, -1)
        y = y.reshape(-1, 1) * omega
        x = x.reshape(-1, 1) * omega
        encoding = torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=-1)
        encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding, persistent=False)
    def forward(self, x):
        encoding = self.encoding.to(x.dtype)
        return x + encoding

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=768 * 4,
            dropout=0,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.encoder = nn.Sequential(
            PositionEncoder(20, 20, 768),
            nn.TransformerEncoder(encoder_layer, num_layers=8, enable_nested_tensor=False),
            nn.LayerNorm(768)
        )
    def forward(self, x):
        return self.encoder(x)

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_tokens = nn.Conv2d(512, 768, 1)
        self.patch_embed = nn.Conv2d(768, 768, 2, 2, bias=False)
        self.vit = ViTEncoder()
        self.patch_unembed = nn.ConvTranspose2d(768, 768, 2, 2, bias=False)
        self.to_features = nn.Conv2d(768, 1024, 1)
    def forward(self, x):
        x = self.to_tokens(x)
        x = self.patch_embed(x)

        b, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)

        x = self.vit(x)

        x = x.transpose(1, 2).reshape(b, 768, h, w)

        x = self.patch_unembed(x)
        x = self.to_features(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2)
        ])
        self.blocks = nn.ModuleList([
            Block(1024, 512),
            Block(512, 256),
            Block(256, 128),
            Block(128, 64)
        ])
    def forward(self, x, encoder_features):
        features = encoder_features[::-1]
        for feature, upsample, block in zip(features, self.upsample, self.blocks):
            x = upsample(x)
            x = torch.cat([feature, x], dim=1)
            x = block(x)
        return x

class TransUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()
        self.classification = nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x, encoder_features = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, encoder_features)
        x = self.classification(x)
        return x

if __name__ == '__main__':
    from src.config import load_config

    config = load_config('configs/transunet.toml')

    model = TransUNet(**config.model.parameters).to(config.device)

    images = torch.zeros(config.batch_size, 1, 640, 640, device=config.device)

    with torch.autocast(config.device):
        predicts = model(images)

    print(f'{tuple(images.shape)} --{model.__class__.__name__}-> {tuple(predicts.shape)}')
