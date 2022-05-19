import torch
from torchvision import models

def convBlock(ni, no):

    return torch.nn.Sequential(
        torch.nn.Conv2d(ni, no, kernel_size=3, padding=1, bias=False),
        torch.nn.BatchNorm2d(no),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

# Contrastive Loss Model

class SiameseNetwork(torch.nn.Module):

    def __init__(self, dim_output=64):
        super(SiameseNetwork, self).__init__()
        self.conv1 = convBlock(3, 16)
        self.conv2 = convBlock(16, 128)
        self.fc1 = torch.nn.Linear(128*25*25, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, dim_output)

    def forward(self, input1, input2):
        x1, y1 = self.conv1(input1), self.conv1(input2)
        x2, y2 = self.conv2(x1), self.conv2(y1)
        x3, y3 = torch.flatten(x2, start_dim=1), torch.flatten(y2, start_dim=1)
        x4, y4 = torch.relu(self.fc1(x3)), torch.relu(self.fc1(y3))
        x5, y5 = torch.relu(self.fc2(x4)), torch.relu(self.fc2(y4))
        x6, y6 = self.fc3(x5), self.fc3(y5)
        output1 = [x3, x4, x5, x6]
        output2 = [y3, y4, y5, y6]

        return output1, output2


def convVGG16():
    vgg16 = models.vgg16(pretrained=True)
    
    for param in vgg16.parameters():
        param.requires_grad = False

    return torch.nn.Sequential(
        vgg16.features,
        #vgg16.avgpool
    )

# Contrastive Loss Model with VGG16

class SiameseNetworkVGG16(torch.nn.Module):

    def __init__(self):
        super(SiameseNetworkVGG16, self).__init__()
        self.conv = convVGG16()
        #self.fc1 = torch.nn.Linear(128*25*25, 512)
        #self.fc1 = torch.nn.Linear(512*7*7, 512)
        self.fc1 = torch.nn.Linear(512*3*3, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 64)

    def forward(self, input1, input2):
        x1, y1 = self.conv(input1), self.conv(input2)
        x2, y2 = torch.flatten(x1, start_dim=1), torch.flatten(y1, start_dim=1)
        x3, y3 = torch.relu(self.fc1(x2)), torch.relu(self.fc1(y2))
        x4, y4 = torch.relu(self.fc2(x3)), torch.relu(self.fc2(y3))
        x5, y5 = self.fc3(x4), self.fc3(y4)
        output1 = [x2, x3, x4, x5]
        output2 = [y2, y3, y4, y5]

        return output1, output2


def convResNet18():
    resnet18 = models.resnet18(pretrained=True)
        
    for param in resnet18.parameters():
        param.requires_grad = False

    return torch.nn.Sequential(
        resnet18.conv1,
        resnet18.bn1,
        resnet18.relu,
        resnet18.maxpool,
        resnet18.layer1,
        resnet18.layer2,
        resnet18.layer3,
        resnet18.layer4,
        resnet18.avgpool
    )

# Contrastive Loss Model with ResNet18

class SiameseNetworkResNet18(torch.nn.Module):

    def __init__(self):
        super(SiameseNetworkResNet18, self).__init__()
        self.conv = convResNet18()
        #self.fc1 = torch.nn.Linear(128*25*25, 512)
        #self.fc1 = torch.nn.Linear(512*7*7, 512)
        self.fc1 = torch.nn.Linear(512*1*1, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 64)

    def forward(self, input1, input2):
        x1, y1 = self.conv(input1), self.conv(input2)
        x2, y2 = torch.flatten(x1, start_dim=1), torch.flatten(y1, start_dim=1)
        x3, y3 = torch.relu(self.fc1(x2)), torch.relu(self.fc1(y2))
        x4, y4 = torch.relu(self.fc2(x3)), torch.relu(self.fc2(y3))
        x5, y5 = self.fc3(x4), self.fc3(y4)
        output1 = [x2, x3, x4, x5]
        output2 = [y2, y3, y4, y5]

        return output1, output2

# Triplet Loss Model

class SiameseNetworkTriplet(torch.nn.Module):

    def __init__(self, dim_output=64):
        super(SiameseNetworkTriplet, self).__init__()
        self.conv1 = convBlock(3, 16)
        self.conv2 = convBlock(16, 128)
        self.fc1 = torch.nn.Linear(128*25*25, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, dim_output)

    def forward(self, input1, input2, input3):
        x1, y1, z1 = self.conv1(input1), self.conv1(input2), self.conv1(input3)
        x2, y2, z2 = self.conv2(x1), self.conv2(y1), self.conv2(z1)
        x3, y3, z3 = torch.flatten(x2, start_dim=1), torch.flatten(y2, start_dim=1), torch.flatten(z2, start_dim=1)
        x4, y4, z4 = torch.relu(self.fc1(x3)), torch.relu(self.fc1(y3)), torch.relu(self.fc1(z3))
        x5, y5, z5 = torch.relu(self.fc2(x4)), torch.relu(self.fc2(y4)), torch.relu(self.fc2(z4))
        x6, y6, z6 = self.fc3(x5), self.fc3(y5), self.fc3(z5)
        output1 = [x3, x4, x5, x6]
        output2 = [y3, y4, y5, y6]
        output3 = [z3, z4, z5, z6]

        return output1, output2, output3

# Quadruplet Loss Model

class SiameseNetworkQuadruplet(torch.nn.Module):

    def __init__(self, dim_output=64):
        super(SiameseNetworkQuadruplet, self).__init__()
        self.conv1 = convBlock(3, 16)
        self.conv2 = convBlock(16, 128)
        self.fc1 = torch.nn.Linear(128*25*25, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, dim_output)

    def forward(self, input1, input2, input3, input4):
        x1, y1, z1, w1 = self.conv1(input1), self.conv1(input2), \
                            self.conv1(input3), self.conv1(input4)
        x2, y2, z2, w2 = self.conv2(x1), self.conv2(y1), \
                            self.conv2(z1), self.conv2(w1)
        x3, y3, z3, w3 = torch.flatten(x2, start_dim=1), torch.flatten(y2, start_dim=1), \
                            torch.flatten(z2, start_dim=1), torch.flatten(w2, start_dim=1)
        x4, y4, z4, w4 = torch.relu(self.fc1(x3)), torch.relu(self.fc1(y3)), \
                            torch.relu(self.fc1(z3)), torch.relu(self.fc1(w3))
        x5, y5, z5, w5 = torch.relu(self.fc2(x4)), torch.relu(self.fc2(y4)), \
                            torch.relu(self.fc2(z4)), torch.relu(self.fc2(w4))
        x6, y6, z6, w6 = self.fc3(x5), self.fc3(y5), \
                            self.fc3(z5), self.fc3(w5)
        output1 = [x3, x4, x5, x6]
        output2 = [y3, y4, y5, y6]
        output3 = [z3, z4, z5, z6]
        output4 = [w3, w4, w5, w6]

        return output1, output2, output3, output4