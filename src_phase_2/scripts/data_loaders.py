from torchvision import transforms
from torch_snippets import Dataset, read
import os
import numpy as np

class BasicTransformations:
    """Rotate by one of the given angles."""

    def __init__(self, image_size=[300, 300], affine_degrees=5, 
            affine_translate=(0.01, 0.02), affine_scale=(0.9, 1.1)
        ):
        self.image_size = image_size
        self.affine_degrees = affine_degrees
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale

    def get_transformations(self):
        transformations_composition = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(self.affine_degrees, self.affine_translate,
                                    scale=self.affine_scale),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        return transformations_composition

class BasicDataset(Dataset):
    def __init__(self, images_folder, compare_file, transform = None, mode=None):
        self.transform = transform
        self.mode = mode

        with open(compare_file, "r") as file:
            lines = file.read().splitlines()
        self.pairs = list(
            map(lambda line : line.split(" "), lines)
        )

        self.images_folder = images_folder

    def __getitem__(self,ix):
        image_1 = self.pairs[ix][0]
        image_2 = self.pairs[ix][1]
        person_1 = image_1.split("_")[0]
        person_2 = image_2.split("_")[0]

        true_label = 1 if person_1 == person_2 else 0
        image_1 = read(
            os.path.join(self.images_folder, image_1), mode=self.mode
        )
        image_2 = read(
            os.path.join(self.images_folder, image_2), mode=self.mode
        )
        if not self.mode:
            image_1 = np.expand_dims(image_1, 2)
            image_2 = np.expand_dims(image_2, 2)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2, np.array([true_label])

    def __len__(self):
        return len(self.pairs)