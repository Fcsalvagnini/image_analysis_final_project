#!pip install -q torch_snippets
from torch_snippets import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch
from siameseNN      import * 
from torchvision    import transforms
import sys

@torch.no_grad()
def validate_batch(model, data, criterion):
    imgsA, imgsB, labels = [t.to(device) for t in data]
    codesA, codesB       = model(imgsA, imgsB)
    loss, acc            = criterion(codesA, codesB, labels)
    return loss.item(), acc.item()

if (len(sys.argv)!=4):
    print("python eval_FPNet.py <P1> <P2> <P3>")
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

dataloader    = DataLoader(dataset,batch_size=32,shuffle=True)
criterion     = ContrastiveLoss()

n_epochs = 200
log      = Report(n_epochs)

mean_loss = 0.0
mean_acc  = 0.0

for epoch in range(n_epochs):
    N = len(dataloader)
    for i, data in enumerate(dataloader):
        loss, acc  = validate_batch(model, data, criterion)
        mean_loss += loss
        mean_acc  += acc
        log.record(epoch+(1+i)/N, val_loss=loss, val_acc=acc, end='\r')
    if (epoch+1)%20==0: log.report_avgs(epoch+1)

log.plot_epochs(['val_loss'], log=True, title = "Loss")
log.plot_epochs(['val_acc'], title = "Accuracy")     

mean_loss = mean_loss / (N*n_epochs)
mean_acc  = mean_acc / (N*n_epochs)

print("loss: ",mean_loss,"acc: ",mean_acc)
