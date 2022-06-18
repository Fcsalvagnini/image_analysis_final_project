import torch
import colorcet as cc
import seaborn as sns
from torch_snippets import *
from sklearn.manifold import TSNE


def train_batch(model, data, optimizer, criterion, device):
    imgsA, imgsB, labels = [t.to(device) for t in data]
    optimizer.zero_grad()
    outputA, outputB = model(imgsA, imgsB)
    codesA, codesB = outputA[-1], outputB[-1]
    loss, acc = criterion(codesA, codesB, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch(model, data, criterion, device):
    imgsA, imgsB, labels = [t.to(device) for t in data]
    outputA, outputB = model(imgsA, imgsB)
    codesA, codesB = outputA[-1], outputB[-1]
    loss, acc = criterion(codesA, codesB, labels)
    return loss.item(), acc.item()


def train_batch_triplet(model, data, optimizer, criterion, device):
    imgsA, imgsB, imgsC = [t.to(device) for t in data]
    optimizer.zero_grad()
    outputA, outputB, outputC = model(imgsA, imgsB, imgsC)
    codesA, codesB, codesC = outputA[-1], outputB[-1], outputC[-1]
    loss, acc = criterion(codesA, codesB, codesC)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch_triplet(model, data, criterion, device):
    imgsA, imgsB, imgsC = [t.to(device) for t in data]
    outputA, outputB, outputC = model(imgsA, imgsB, imgsC)
    codesA, codesB, codesC = outputA[-1], outputB[-1], outputC[-1]
    loss, acc = criterion(codesA, codesB, codesC)
    return loss.item(), acc.item()


def train_batch_quadruplet(model, data, optimizer, criterion, device):
    imgsA, imgsB, imgsC, imgsD = [t.to(device) for t in data]
    optimizer.zero_grad()
    outputA, outputB, outputC, outputD = model(imgsA, imgsB, imgsC, imgsD)
    codesA, codesB, codesC, codesD = outputA[-1], outputB[-1], outputC[-1], outputD[-1]
    loss, acc = criterion(codesA, codesB, codesC, codesD)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch_quadruplet(model, data, criterion, device):
    imgsA, imgsB, imgsC, imgsD = [t.to(device) for t in data]
    outputA, outputB, outputC, outputD = model(imgsA, imgsB, imgsC, imgsD)
    codesA, codesB, codesC, codesD = outputA[-1], outputB[-1], outputC[-1], outputD[-1]
    loss, acc = criterion(codesA, codesB, codesC, codesD)
    return loss.item(), acc.item()


def get_classifier_model(sinn_model, nclasses, device):
    
    for param in sinn_model.parameters():
        param.requires_grad = False
    
    output = nclasses
    model = torch.nn.Sequential(sinn_model.conv1,
                                sinn_model.conv2,
                                torch.nn.Flatten(), \
                                torch.nn.Linear(128*25*25,128), \
                                torch.nn.ReLU(), \
                                torch.nn.Dropout(0.2), \
                                torch.nn.Linear(128, output))
    loss_fn   = torch.nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return model.to(device), loss_fn, optimizer


def train_batch_classifier(batch, targets, model, optimizer, loss_fn):
    model.train() 
    prediction = model(batch) 
    batch_loss = loss_fn(prediction, targets) 
    batch_loss.backward() 
    optimizer.step() 
    optimizer.zero_grad()

    return batch_loss.detach().cpu()


def validate_batch_classifier(batch, targets, model, loss_fn):
    model.eval() 
    prediction = model(batch) 
    batch_loss = loss_fn(prediction, targets) 
    return batch_loss.detach().cpu()


def get_accuracy_classifier(batch, targets, model, device):
    model.eval() 
    predictions = model(batch) 
    
    accuracy = (torch.max(predictions, dim=1)[1] == targets).sum() * 1.0 / len(targets)

    return accuracy.item()


def save_figure(training_list, validation_list, title, xlabel, ylabel, filepath):
    plt.figure(figsize=(8,5))
    plt.title(title)
    plt.plot(training_list, label="train")
    plt.plot(validation_list, label="val", linestyle='dashed')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(filepath)
    plt.clf()


def save_tsne_projection(output, labels, layer, num_classes, title, filepath):
    tsne = TSNE(n_components=2, perplexity=40, init='random', learning_rate='auto')
    projection = tsne.fit_transform(output[layer].detach().cpu().numpy())
    df = pd.DataFrame()
    df["y"] = labels.detach().cpu().numpy()
    df["x1"] = projection[:, 0]
    df["x2"] = projection[:, 1]

    palette = sns.color_palette(cc.glasbey, n_colors=num_classes)
    sns.scatterplot(x='x1', y='x2', hue=df.y.tolist(),
            palette=palette,
            data=df, s=25, legend=False).set(title=title)

    plt.savefig(filepath)
    plt.clf()