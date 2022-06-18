import torch
import seaborn as sns
import colorcet as cc
from torchvision import models
from torchvision import transforms
from torch_snippets import *
from utils import *
from models_corel import *
from datasets_corel import *
from losses import *
from random import shuffle, seed
from sklearn.manifold import TSNE

if len(sys.argv) != 2:
    print('Usage: python3 corel_transfer.py model_name.pth')
    exit(0)

model_file = sys.argv[1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torchvision import transforms

folder   = './images/corel/'
file_ext = "*.png"  
nclasses = 6 

train_perc = 0.40
valid_perc = 0.20
test_perc = 0.40

fileset = Glob(folder + file_ext) # it returns a list of filenames
num_train_samples = int(len(fileset)*train_perc)
num_valid_samples = int(len(fileset)*valid_perc)  
num_test_samples  = len(fileset) - num_train_samples - num_valid_samples 

seed(10)
shuffle(fileset)

trainset = fileset[:num_train_samples]
validset = fileset[num_train_samples:num_train_samples+num_valid_samples]
testset  = fileset[num_train_samples+num_valid_samples:]

# Data augmentation performed using transforms.RandomHorizontalFlip and transforms.RandomAffine
trn_tfms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(5, (0.01, 0.2),
                            scale=(0.9, 1.1)),
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Data augmentation is only performed in the training set
val_tfms = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trn_ds = StandardDataset(items=trainset, transform=trn_tfms)
val_ds = StandardDataset(items=validset, transform=val_tfms)
tst_ds = StandardDataset(items=testset, transform=val_tfms)
tot_ds = StandardDataset(items=fileset, transform=val_tfms)

trn_dl = DataLoader(trn_ds, shuffle=True, batch_size=32)
val_dl = DataLoader(val_ds, shuffle=True, batch_size=32)
tst_dl = DataLoader(tst_ds, shuffle=True, batch_size=32)
tot_dl = DataLoader(tot_ds, shuffle=True, batch_size=len(tot_ds))

sinn_model = torch.load(model_file)

model, loss_fn, optimizer = get_classifier_model(sinn_model, nclasses, device)

nepochs = 40
for epoch in range(nepochs):
    n = len(trn_dl)
    for ix, data in enumerate(trn_dl):
        batch, targets = data
        batch = batch.to(device)
        targets = targets.to(device)
        targets = targets.squeeze()

        batch_loss = train_batch_classifier(batch, targets, model, optimizer, loss_fn)        

    model.eval()
    trn_iter_dl = DataLoader(trn_ds, shuffle=True, batch_size=len(trn_ds))  
    batch, targets = next(iter(trn_iter_dl))
    batch, targets = batch.to(device), targets.to(device)
    trn_accuracy = get_accuracy_classifier(batch, targets, model, device)

    val_iter_dl = DataLoader(val_ds, shuffle=True, batch_size=len(val_ds))  
    batch, targets = next(iter(val_iter_dl))
    batch, targets = batch.to(device), targets.to(device)
    val_accuracy = get_accuracy_classifier(batch, targets, model, device)
     
    print('Epoch', epoch + 1, 'trn_acc:', trn_accuracy, 'val_acc:', val_accuracy, '\r')

tst_iter_dl = DataLoader(tst_ds, shuffle=True, batch_size=len(val_ds))  
batch, targets = next(iter(tst_iter_dl))
batch, targets = batch.to(device), targets.to(device)
tst_accuracy = get_accuracy_classifier(batch, targets, model, device)

batch_tot, targets_tot = next(iter(tot_dl))
batch_tot, targets_tot = batch_tot.to(device), targets_tot.to(device)
save_tsne_projection([ model(batch_tot) ], targets_tot, 0, nclasses, "Projection of the output MLP layer", "classifier_contrastive_corel_lstar.pdf")

print('tst_acc:', tst_accuracy)