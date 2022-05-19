from torch_snippets import *


class StandardDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
    def __getitem__(self, ix): # returns two images of the same subject if same_person is True
        item = self.items[ix]
        label = int(fname(item).split('_')[0])
        img = Image.open(item)
        if self.transform:
            img = self.transform(img)
        return img, label - 1
    def __len__(self):
        return len(self.items)


class SiameseNetworkDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
    def __getitem__(self, ix): # returns two images of the same subject if same_person is True
        itemA = self.items[ix]
        label = int(fname(itemA).split('_')[0])
        same_label = randint(2)
        if same_label:
            while True:
                itemB = choose(self.items)
                if label == int(fname(itemB).split('_')[0]):
                    break
        else:
            while True:
                itemB = choose(self.items)
                if label != int(fname(itemB).split('_')[0]):
                    break
        imgA = Image.open(itemA)
        imgB = Image.open(itemB)
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB, np.array([1-same_label])
    def __len__(self):
        return len(self.items)


class SiameseNetworkDatasetTriplet(Dataset):
    def __init__(self, items, transform=None, should_invert=True):
        self.items = items
        self.transform = transform
    def __getitem__(self, ix): # returns two images of the same subject if same_person is True
        itemA = self.items[ix]
        label = int(fname(itemA).split('_')[0])
        while True:
            itemB = choose(self.items)
            if label == int(fname(itemB).split('_')[0]) and itemA != itemB:
                break
        while True:
            itemC = choose(self.items)
            if label != int(fname(itemC).split('_')[0]):
                break
        imgA = Image.open(itemA)
        imgB = Image.open(itemB)
        imgC = Image.open(itemC)
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
            imgC = self.transform(imgC)
        return imgA, imgB, imgC

    def __len__(self):
        return len(self.items)


class SiameseNetworkDatasetQuadruplet(Dataset):
    def __init__(self, items, transform=None, should_invert=True):
        self.items = items
        self.transform = transform
    def __getitem__(self, ix): # returns two images of the same subject if same_person is True
        itemA = self.items[ix]
        label = int(fname(itemA).split('_')[0])
        while True:
            itemB = choose(self.items)
            if label == int(fname(itemB).split('_')[0]) and itemA != itemB:
                break
        while True:
            itemC = choose(self.items)
            if label != int(fname(itemC).split('_')[0]):
                break
        while True:
            itemD = choose(self.items)
            if label != int(fname(itemD).split('_')[0]) and int(fname(itemC).split('_')[0]) != int(fname(itemD).split('_')[0]):
                break
        imgA = Image.open(itemA)
        imgB = Image.open(itemB)
        imgC = Image.open(itemC)
        imgD = Image.open(itemD)
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
            imgC = self.transform(imgC)
            imgD = self.transform(imgD)
        return imgA, imgB, imgC, imgD

    def __len__(self):
        return len(self.items)

