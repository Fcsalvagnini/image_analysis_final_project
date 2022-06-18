#!pip install -q torch_snippets
from torch_snippets import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch
from siameseNN      import * 
from torchvision    import transforms
import sys

if (len(sys.argv)!=4):
    print("python train_FPNet.py <P1> <P2> <P3>")
    print("P1: folder with images of a dataset")
    print("P2: csv file with image comparisons for training")
    print("P3: model.pth")
    exit()
    
trn_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(5, (0.01,0.2),
                            scale=(0.9,1.1)),
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

dataset    = SiameseNetworkDataset(image_folder=sys.argv[1], imagenames_file = sys.argv[2], transform=trn_tfms)

#show(dataset[10][0].permute(1,2,0).numpy())
#show(dataset[10][1].permute(1,2,0).numpy())

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model      = SiameseNetwork().to(device)
criterion  = ContrastiveLoss()
optimizer  = optim.Adam(model.parameters(),lr = 0.001, weight_decay=0.01)
n_epochs   = 200
log        = Report(n_epochs)

for epoch in range(n_epochs):
    N = len(dataloader)
    for i, data in enumerate(dataloader):
        loss, acc = train_batch(model, data, optimizer, criterion)
        log.record(epoch+(1+i)/N, trn_loss=loss, trn_acc=acc, end='\r')
    if (epoch+1)%20==0: log.report_avgs(epoch+1)

log.plot_epochs(['trn_loss'], log=True, title = "Loss")
log.plot_epochs(['trn_acc'], title = "Accuracy")     

torch.save(model, sys.argv[3])
