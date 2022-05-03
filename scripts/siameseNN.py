#!pip install -q torch_snippets
from torch_snippets import *

contrastive_thres = 1.1

class SiameseNetworkDataset(Dataset):
    def __init__(self, image_folder, imagenames_file, transform=None):
        self.image_folder    = image_folder
        self.imagenames_file = imagenames_file
        self.nitems          = 0
        self.transform       = transform
    def __getitem__(self, ix):
        f     = open(self.imagenames_file,"r")
        pairs = []
        for line in f:
            pair = line.strip().split(" ")
            pairs.append(pair)

        self.nitems = len(pairs)    

        image1      = pairs[ix][0]
        image2      = pairs[ix][1]
        
        person1 = image1.split("_")[0]
        person2 = image2.split("_")[0]
        
        if (person1 == person2):
            truelabel = 0
        else:
            truelabel = 1
        
        image1 = read("{}/{}".format(self.image_folder,image1))
        image2 = read("{}/{}".format(self.image_folder,image2))
        image1 = np.expand_dims(image1,2)
        image2 = np.expand_dims(image2,2)
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, np.array([truelabel])
    def __len__(self):
        f     = open(self.imagenames_file,"r")
        pairs = []
        for line in f:
            pair = line.strip().split(" ")
            pairs.append(pair)

        self.nitems = len(pairs)    
        return (self.nitems)

    
def convBlock(ni, no):
    return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size=3, padding=1, bias=False), #, padding_mode='reflect'),
        nn.BatchNorm2d(no),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.features = nn.Sequential(
            convBlock(1,16),
            convBlock(16,128),
            nn.Flatten(),
            nn.Linear(128*25*25, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )

    def forward(self, input1, input2):
        output1 = self.features(input1)
        output2 = self.features(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2)/2 +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))/2
        acc = ((euclidean_distance > contrastive_thres) == label).float().mean()
        return loss_contrastive, acc


def train_batch(model, data, optimizer, criterion):
    imgsA, imgsB, labels = [t.to(device) for t in data]
    optimizer.zero_grad()
    codesA, codesB = model(imgsA, imgsB)
    loss, acc = criterion(codesA, codesB, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()
    
