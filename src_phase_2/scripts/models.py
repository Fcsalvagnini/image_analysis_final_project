from torch_snippets import nn

def ConvBlock(channels_in, channels_out, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 
                    kernel_size=kernel_size, padding=kernel_size//2,
                    bias=False
                ),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1
        )
    )

class SimpleConvSiameseNN(nn.Module):
    def __init__(self):
        super(SimpleConvSiameseNN, self).__init__()
        
        self.features = nn.Sequential(
            ConvBlock(channels_in=1, channels_out=128),
            ConvBlock(channels_in=128, channels_out=64),
            ConvBlock(channels_in=64, channels_out=32),
            nn.Flatten(),
            nn.Linear(32*15*15, 512), 
            nn.ReLU(inplace=True),
            nn.Linear(512, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )
    
    def forward(self, input_1, input_2):
        output_1 = self.features(input_1)
        output_2 = self.features(input_2)

        return output_1, output_2