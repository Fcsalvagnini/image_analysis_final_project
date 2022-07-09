import torch
import timm
from torch_snippets import nn
from vit_transformer import PatchEmbedding, TransformerEncoder
from torchvision import models
from einops.layers.torch import Reduce
import math

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


class ViTSiameseTriplet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 patch_size: int,
                 emb_size: int,
                 img_size: int,
                 depth: int,
                 **kwargs):
        super(ViTSiameseTriplet, self).__init__()

        self.vit_transformer = ViT(in_channels,
                                   patch_size,
                                   emb_size,
                                   img_size,
                                   depth,
                                   **kwargs)

    def forward(self, input_anchor, input_pos, input_neg):
        output_anchor = self.vit_transformer(input_anchor)
        output_pos = self.vit_transformer(input_pos)
        output_neg = self.vit_transformer(input_neg)

        return output_anchor, output_pos, output_neg


def ConvBlock(
        channels_in, channels_out, kernel_size=3, padding=False, use_bn=True,
        activation_function=None, pool=False
    ):
    if padding:
        padding = "same"
    else:
        padding = "valid"
    bias = False if use_bn else True
    op_list = [
        nn.Conv2d(channels_in, channels_out,
                  kernel_size=kernel_size, padding=padding,
                  bias=bias
                  )
    ]
    if use_bn:
        op_list.append(nn.BatchNorm2d(channels_out))
    if activation_function:
        op_list.append(activation_function)
    if pool:
        op_list.append(
            nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)
        )
    
    return nn.Sequential(*op_list)


# Implement Siamese based on the one proposed in the Constrastive Loss Paper

# Network based on the "Siamese Network Based on CNN for Fingerprint Recognition" paper
class ConvFingerprintSiamese(nn.Module):
    def __init__(self, input_size, activation=False):
        super(ConvFingerprintSiamese, self).__init__()
        self.input_size = input_size
        if activation:
            activation = nn.ReLU(inplace=True)
        else:
            activation = None

        features_layers_extractors = [
            nn.ReflectionPad2d(1),
            ConvBlock(
                channels_in=1, channels_out=4, padding=False,
                activation_function=activation
            ),
            nn.ReflectionPad2d(1),
            ConvBlock(
                channels_in=4, channels_out=8, padding=False,
                activation_function=activation
            ),
            nn.ReflectionPad2d(1),
            ConvBlock(
                channels_in=8, channels_out=8, padding=False,
                activation_function=activation
            ),
            nn.Flatten(),
        ]

        linear_layers = [
            nn.Linear(input_size[0] * input_size[1] * 8, 500),
            nn.Linear(500,500),
            nn.Linear(500,5)
        ]

        for idx in range(len(linear_layers)):
            features_layers_extractors.append(linear_layers[idx])
            if activation and idx != len(linear_layers) - 1:
                features_layers_extractors.append(activation)

        self.features = nn.Sequential(*features_layers_extractors)

    def forward(self, input_1, input_2):
        output_1 = self.features(input_1)
        output_2 = self.features(input_2)

        return output_1, output_2

    def _initialize_weights(self):
        #for each submodule of our network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #get the number of elements in the layer weights
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels    
                #initialize layer weights with random values generated from a normal
                #distribution with mean = 0 and std = sqrt(2. / n))
                m.weight.data.normal_(mean=0, std=math.sqrt(2. / n))

                if m.bias is not None:
                    #initialize bias with 0 
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #initialize layer weights with random values generated from a normal
                #distribution with mean = 0 and std = 1/100
                m.weight.data.normal_(mean=0, std=0.01)
                if m.bias is not None:
                #initialize bias with 0 
                    m.bias.data.zero_()

class SimpleConvSiameseNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleConvSiameseNN, self).__init__()
        self.input_size = input_size

        self.features = nn.Sequential(
            ConvBlock(
                channels_in=1, channels_out=128, padding=True, 
                activation_function=nn.ReLU(inplace=True), pool=True
            ),
            ConvBlock(
                channels_in=128, channels_out=64, padding=True, 
                activation_function=nn.ReLU(inplace=True), pool=True
            ),
            ConvBlock(
                channels_in=64, channels_out=32, padding=True, 
                activation_function=nn.ReLU(inplace=True), pool=True
            ),
            nn.Flatten(),
            nn.Linear(128 * (self.input_size[0] // 8) * (self.input_size[1] // 8), 512),
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

def get_n_out_features(encoder, img_size, nchannels):
    out_feature = encoder(torch.randn(1, nchannels, img_size, img_size))
    n_out = 1
    for dim in out_feature[-1].shape:
        n_out *= dim
    return n_out

class SiameseNetworkTimmBackbone(nn.Module):
    def __init__(self, network:str, image_size:int, nchannels: int, transformers: bool=False):
        super().__init__()
        if transformers:
            model_creator = {'model_name': network,
                             "pretrained":True,
                             "num_classes": 0}
        else:
            model_creator = {'model_name': network,
                             "pretrained":True,
                             "features_only": True}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = False
        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        if transformers:
            self.dimensionality_reductor = nn.Sequential(
                nn.Linear(n_out, 512), nn.ReLU(inplace = True),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 64)
            )
        else:
            self.dimensionality_reductor = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(n_out, 512), nn.ReLU(inplace = True),
                        nn.Linear(512, 256), nn.ReLU(inplace=True),
                        nn.Linear(256, 64)
            )


    def forward(self, input1, input2):
        output1 = self.encoder(input1)[-1]
        output1 = self.dimensionality_reductor(output1)
        output2 = self.encoder(input2)[-1]
        output2 = self.dimensionality_reductor(output2)

        return output1, output2