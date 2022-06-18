import torch
import seaborn as sns
import colorcet as cc
from torchvision import models
from torchvision import transforms
from torch_snippets import *
from utils import *
from models_faces import *
from datasets_faces import *
from losses import *
from sklearn.manifold import TSNE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data augmentation performed using transforms.RandomHorizontalFlip and transforms.RandomAffine
trn_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(5, (0.01, 0.2),
                            scale=(0.9, 1.1)),
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Data augmentation is only performed in the training set
val_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trn_ds = SiameseNetworkDatasetTriplet(folder="./images/faces/training", transform=trn_tfms)
val_ds = SiameseNetworkDatasetTriplet(folder="./images/faces/testing", transform=val_tfms)
tot_ds = StandardDataset(folder="./images/faces/total", transform=val_tfms)

trn_dl = DataLoader(trn_ds, shuffle=True, batch_size=32)
val_dl = DataLoader(val_ds, shuffle=True, batch_size=32)
tot_dl = DataLoader(tot_ds, shuffle=True, batch_size=len(tot_ds))

trn_iter_dl = DataLoader(trn_ds, shuffle=True, batch_size=len(trn_ds))
val_iter_dl = DataLoader(val_ds, shuffle=True, batch_size=len(val_ds))

nclasses = 40

model = SiameseNetworkTriplet().to(device)
criterion = TripletLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001, weight_decay=0.01)

n_epochs = 200
log = Report(n_epochs)
training_acc = []
training_loss = []
validation_acc = []
validation_loss = []

for epoch in range(n_epochs):
    N = len(trn_dl)
    model.train()
    for i, data in enumerate(trn_dl):
        loss, acc = train_batch_triplet(model, data, optimizer, criterion, device)
        log.record(epoch+(1+i)/N, trn_loss=loss, trn_acc=acc, end='\r')
        
    N = len(val_dl)
    model.eval()
    for i, data in enumerate(val_dl):
        loss, acc = validate_batch_triplet(model, data, criterion, device)
        log.record(epoch+(1+i)/N, val_loss=loss, val_acc=acc, end='\r')
        
    trn_batch_dl = next(iter(trn_iter_dl))
    loss, acc = validate_batch_triplet(model, trn_batch_dl, criterion, device)

    training_loss.append(loss)
    training_acc.append(acc)

    val_batch_dl = next(iter(val_iter_dl))
    loss, acc = validate_batch_triplet(model, val_batch_dl, criterion, device)
    
    validation_loss.append(loss)
    validation_acc.append(acc)
    
    if (epoch+1)%20==0: log.report_avgs(epoch+1)

trn_batch_dl = next(iter(trn_iter_dl))
loss, acc = validate_batch_triplet(model, trn_batch_dl, criterion, device)
print('Training Loss:', loss, 'Training Accuracy:', acc)

val_batch_dl = next(iter(val_iter_dl))
loss, acc = validate_batch_triplet(model, val_batch_dl, criterion, device)
print('Validation Loss:', loss, 'Validation Accuracy:', acc)

print('Saving Loss and Accuracy curves')

save_figure(training_loss, validation_loss, "Training and Validation Loss", "Epochs", "Loss", "loss_triplet_faces.pdf")

save_figure(training_acc, validation_acc, "Training and Validation Accuracy", "Epochs", "Accuracy", "accuracy_triplet_faces.pdf")

tot_dl = next(iter(tot_dl))
imgs, labels = [t.to(device) for t in tot_dl]

outputA, outputB, outputC = model(imgs, imgs, imgs)

print('Saving t-SNE projections')

save_tsne_projection(outputA, labels, 0, nclasses, "Projection of the input MLP layer", "tsne-1st-layer_triplet_faces.pdf")

save_tsne_projection(outputA, labels, 1, nclasses, "Projection of the first hidden MLP layer", "tsne-2nd-layer_triplet_faces.pdf")

save_tsne_projection(outputA, labels, 2, nclasses, "Projection of the second hidden MLP layer", "tsne-3rd-layer_triplet_faces.pdf")

save_tsne_projection(outputA, labels, 3, nclasses, "Projection of the output MLP layer", "tsne-4th-layer_triplet_faces.pdf")

    


