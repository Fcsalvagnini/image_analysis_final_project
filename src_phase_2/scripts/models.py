from torch_snippets import nn
from vit_transformer import PatchEmbedding, TransformerEncoder
from torchvision import models
from einops.layers.torch import Reduce


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 patch_size: int,
                 emb_size: int,
                 img_size: int,
                 depth: int,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size)
        )


class ViTSiamese(nn.Module):
    def __init__(self,
                 in_channels: int,
                 patch_size: int,
                 emb_size: int,
                 img_size: int,
                 depth: int,
                 **kwargs):
        super(ViTSiamese, self).__init__()

        self.vit_transformer = ViT(in_channels,
                                   patch_size,
                                   emb_size,
                                   img_size,
                                   depth,
                                   **kwargs)

    def forward(self, input_1, input_2):
        output_1 = self.vit_transformer(input_1)
        output_2 = self.vit_transformer(input_2)

        return output_1, output_2


def ConvBlock(channels_in, channels_out, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out,
                  kernel_size=kernel_size, padding=kernel_size // 2,
                  bias=False
                  ),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1
                     )
    )


class SimpleConvSiameseNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleConvSiameseNN, self).__init__()
        self.input_size = input_size

        self.features = nn.Sequential(
            ConvBlock(channels_in=1, channels_out=128),
            ConvBlock(channels_in=128, channels_out=64),
            ConvBlock(channels_in=64, channels_out=32),
            nn.Flatten(),
            nn.Linear(32 * (self.input_size[0] // 8) * (self.input_size[1] // 8), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )

    def forward(self, input_1, input_2):
        output_1 = self.features(input_1)
        output_2 = self.features(input_2)

        return output_1, output_2


class PreTrainedVGGSiameseNN(nn.Module):
    def __init__(self):
        super(PreTrainedVGGSiameseNN, self).__init__()
        self.features = nn.Sequential(
            *list(models.vgg16(pretrained=True).children())[:-1]
        )
        for parameter in self.features.parameters():
            parameter.requires_grad = False

        self.dimensionality_reductor = nn.Sequential(
            nn.Flatten(),
            models.vgg16(pretrained=True).classifier[0],
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
        )

    def forward(self, input_1, input_2):
        output_1 = self.features(input_1)
        output_1 = self.dimensionality_reductor(output_1)
        output_2 = self.features(input_2)
        output_2 = self.dimensionality_reductor(output_2)

        return output_1, output_2
