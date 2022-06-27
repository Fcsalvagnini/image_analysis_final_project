import os

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class FingerprintDataset(Dataset):
    def __init__(self, pairs_df: pd.DataFrame, data_path: str, transforms=None):
        # images dir
        self.data_path = data_path 
        # txt with doublets/trilets/etc
        self.pairs_df = pairs_df
        self.transforms = transforms    

        self.nitems = 0


    def __getitem__(self, idx):
        image1, image2 = self.pairs_df.loc[idx][0], self.pairs_df.loc[idx][1]
        person1, person2 = image1.split("_")[0], image2.split("_")[0]
        if (person1 == person2):
            label = 0
        else:
            label = 1

        image1 = cv2.imread(os.path.join(self.data_path, image1))
        image2 = cv2.imread(os.path.join(self.data_path, image2))

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image1 = self.transforms(image=image1)['image']
            image2 = self.transforms(image=image2)['image']
        #else:
        #    image1 = np.expand_dims(image1, 2)
        #    image2 = np.expand_dims(image2, 2)

        return image1, image2, np.array([label])


    def __len__(self):
        return len(self.pairs_df)
