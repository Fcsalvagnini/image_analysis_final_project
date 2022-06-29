from torchvision import transforms
from torch_snippets import Dataset, read
import os
import numpy as np
import random
from utils import create_triplet

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
    def __init__(self, images_folder, compare_file, transform=None, mode=None):
        self.transform = transform
        self.mode = mode

        with open(compare_file, "r") as file:
            lines = file.read().splitlines()
        self.pairs = list(
            map(lambda line: line.split(" "), lines)
        )

        self.images_folder = images_folder

    def __getitem__(self, ix):
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

class BasicStratifiedDataset(Dataset):
    def __init__(self, images_folder, compare_file, transform=None, mode=None,
                    stratify_each_epoch=False
                ):
        self.images_folder = images_folder
        self.transform = transform
        self.mode = mode
        self.stratify_each_epoch = stratify_each_epoch

        with open(compare_file, "r") as file:
            lines = file.read().splitlines()
        pairs = list(
            map(lambda line: line.split(" "), lines)
        )

        # Get all similar and dissimilar pairs
        self.similar_pairs, self.dissimilar_pairs = self.get_pairs(pairs)
        self.n_similar_pairs = len(self.similar_pairs)

    def stratify_dataset(self):
        pairs = self.similar_pairs + random.sample(
                                            self.dissimilar_pairs,
                                            self.n_similar_pairs
                                        )
        np.random.shuffle(pairs)

        return pairs

    def get_pairs(self, pairs):
        similar_pairs = []
        dissimilar_pairs = []
        for pair in pairs:
            img_1_subject = pair[0].split("_")[0]
            img_2_subject = pair[1].split("_")[0]
            if img_1_subject == img_2_subject:
                similar_pairs.append(pair)
            else:
                dissimilar_pairs.append(pair)

        return similar_pairs, dissimilar_pairs
    
    def __getitem__(self, ix):
        # Stratify and shuffle on first batch ()
        if ix == 0 and self.stratify_each_epoch:
            self.pairs = self.stratify_dataset()

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
        return len(self.similar_pairs) * 2

class BasicDatasetTriplet(Dataset):
    def __init__(self, images_folder, compare_file, transform=None, mode=None):
        self.transform = transform
        self.mode = mode

        self.triplets = create_triplet(compare_file)

        self.images_folder = images_folder

    def __getitem__(self, ix):
        image_anchor = self.triplets[ix][0]
        image_pos = self.triplets[ix][1]
        image_neg = self.triplets[ix][2]
        image_anchor = read(
            os.path.join(self.images_folder, image_anchor), mode=self.mode
        )
        image_pos = read(
            os.path.join(self.images_folder, image_pos), mode=self.mode
        )
        image_neg = read(
            os.path.join(self.images_folder, image_neg), mode=self.mode
        )

        if self.transform:
            image_anchor = self.transform(image_anchor)
            image_pos = self.transform(image_pos)
            image_neg = self.transform(image_neg)

        return image_anchor, image_pos, image_neg

    def __len__(self):
        return len(self.triplets)


if __name__ == "__main__":
    basic_dataset = BasicDatasetTriplet("../data/registred_images_v1_train/",
                                        "../compare_files/compare_splited_v1_train_new.txt")

    # basic_dataset = BasicDataset("../data/registred_images_v1_train/",
    #                                     "../compare_files/compare_splited_v1_train_new.txt")

    print(basic_dataset[0])
