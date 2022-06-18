from torch_snippets import *


class StandardDataset(Dataset):
    def __init__(self, folder, transform=None, should_invert=True):
        self.folder = folder
        self.items = Glob('{}/*/*'.format(self.folder))
        self.transform = transform
    def __getitem__(self, ix): # returns two images of the same subject if same_person is True
        item = self.items[ix]
        person = fname(parent(item))
        label = int(person[1:])
        img = read(item)
        if self.transform:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.items)


class SiameseNetworkDataset(Dataset):
    def __init__(self, folder, transform=None, should_invert=True):
        self.folder = folder
        self.items = Glob('{}/*/*'.format(self.folder))
        self.transform = transform
    def __getitem__(self, ix): # returns two images of the same subject if same_person is True
        itemA = self.items[ix]
        person = fname(parent(itemA))
        same_person = randint(2)
        if same_person:
            itemB = choose(Glob('{}/{}/*'.format(self.folder,person), silent=True))
        else:
            while True:
                itemB = choose(self.items)
                if person != fname(parent(itemB)):
                    break
        imgA = read(itemA)
        imgB = read(itemB)
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB, np.array([1-same_person])
    def __len__(self):
        return len(self.items)


class SiameseNetworkDatasetTriplet(Dataset):
    def __init__(self, folder, transform=None, should_invert=True):
        self.folder = folder
        self.items = Glob('{}/*/*'.format(self.folder))
        self.transform = transform
    def __getitem__(self, ix): # returns two images of the same subject if same_person is True
        itemA = self.items[ix]
        person = fname(parent(itemA))
        while True:
            itemB = choose(self.items)
            if person == fname(parent(itemB)) and itemA != itemB:
                break
        while True:
            itemC = choose(self.items)
            if person != fname(parent(itemC)):
                break
        imgA = read(itemA)
        imgB = read(itemB)
        imgC = read(itemC)
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
            imgC = self.transform(imgC)
        return imgA, imgB, imgC

    def __len__(self):
        return len(self.items)


class SiameseNetworkDatasetQuadruplet(Dataset):
    def __init__(self, folder, transform=None, should_invert=True):
        self.folder = folder
        self.items = Glob('{}/*/*'.format(self.folder))
        self.transform = transform
    def __getitem__(self, ix): # returns two images of the same subject if same_person is True
        itemA = self.items[ix]
        person = fname(parent(itemA))
        while True:
            itemB = choose(self.items)
            if person == fname(parent(itemB)) and itemA != itemB:
                break
        while True:
            itemC = choose(self.items)
            if person != fname(parent(itemC)):
                break
        while True:
            itemD = choose(self.items)
            if person != fname(parent(itemD)) and fname(parent(itemC)) != fname(parent(itemD)):
                break
        imgA = read(itemA)
        imgB = read(itemB)
        imgC = read(itemC)
        imgD = read(itemD)
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
            imgC = self.transform(imgC)
            imgD = self.transform(imgD)
        return imgA, imgB, imgC, imgD

    def __len__(self):
        return len(self.items)

