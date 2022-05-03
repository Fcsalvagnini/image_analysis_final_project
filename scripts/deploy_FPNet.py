#!pip install -q torch_snippets
from torch_snippets import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch
from siameseNN      import * 
from torchvision    import transforms
import sys

if (len(sys.argv)!=4):
    print("python deploy_FPNet.py <P1> <P2> <P3>")
    print("P1: folder with images of a dataset")
    print("P2: csv file with image comparisons for evaluation")
    print("P3: model.pth")
    exit()

model         = torch.load(sys.argv[3], map_location=device)
model.eval()

val_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

dataset       = SiameseNetworkDataset(image_folder=sys.argv[1], imagenames_file = sys.argv[2], transform=val_tfms)

dataloader    = DataLoader(dataset,batch_size=1,shuffle=True)

do_comparison = 'y'

while(do_comparison=='y'):
    dataiter                  = iter(dataloader)
    image1, image2, truelabel = next(dataiter)
    concatenated              = torch.cat((image1*0.5+0.5, image2*0.5+0.5),0)
    output1,output2           = model(image1.to(device),image2.to(device))
    euclidean_distance        = F.pairwise_distance(output1, output2)
    if (euclidean_distance.item() <= contrastive_thres):
        if (truelabel != 0):
            output = 'Same Person, which is an error.'
        else:
            output = 'Same Person, which is correct.'
    else:
        if (truelabel == 0):
            output = 'Different, which is an error.'
        else:
            output = 'Different, which is correct.'
    
    show(torchvision.utils.make_grid(concatenated),
         title='Dissimilarity: {:.2f}\n{}'.format(euclidean_distance.item(), output))
    plt.show()
    do_comparison = input("Type y to continue: ")
    
    
