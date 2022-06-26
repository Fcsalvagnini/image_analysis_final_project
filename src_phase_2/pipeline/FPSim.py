import torch
import torch.nn as nn
from torchvision import models
import timm #https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055


def get_n_out_features(encoder, img_size, nchannels):
    out_feature = encoder(torch.randn(1, nchannels, img_size, img_size))
    n_out = 1
    for dim in out_feature[-1].shape:
        n_out *= dim
    return n_out


class SiameseNetworkTimmBackbone(nn.Module):
    def __init__(self, model_name:str, img_size:int, nchannels: int):
        super().__init__()
        self.encoder = timm.create_model(model_name=model_name, 
                                         pretrained=True,
                                         features_only=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        n_out = get_n_out_features(self.encoder, img_size, nchannels)

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

class SiameseNetworkVGGBackbone(nn.Module):
    def __init__(self):
        super(SiameseNetworkVGGBackbone, self).__init__()
        self.features = nn.Sequential(*list(models.vgg16(pretrained=True).children())[:-1])
        for param in self.features.parameters():
          param.requires_grad = False
      
        self.dimensionality_reductor = nn.Sequential(
            nn.Flatten(),
            models.vgg16(pretrained=True).classifier[0],
            nn.Linear(4096, 512), nn.ReLU(inplace = True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )

    def forward(self, input1, input2):
        
        output1 = self.features(input1)
        output1 = self.dimensionality_reductor(output1)
        output2 = self.features(input2)
        output2 = self.dimensionality_reductor(output2)

        return output1, output2


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.features = nn.Sequential(
            self.convBlock(1,16),
            self.convBlock(16,128),
            nn.Flatten(),
            nn.Linear(128*25*25, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )
    
    def convBlock(self, ni, no):
        return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size=3, padding=1, bias=False), #, padding_mode='reflect'),
        nn.BatchNorm2d(no),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    def forward(self, input1, input2):
        output1 = self.features(input1)
        output2 = self.features(input2)
        return output1, output2

