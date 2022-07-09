from torchvision import transforms
from torch_snippets import Dataset, read
import cv2
import os
from skimage.morphology import skeletonize
import numpy as np
import random

import images_preprocessing as imp

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Perspective,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Affine,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, RandomResizedCrop,
    Sharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, Lambda, ElasticTransform, ImageCompression, ToFloat
)
from albumentations.pytorch import ToTensorV2

from utils import create_triplet, create_pairs_balanced, create_triplets_dir


class BasicTransformations:
    """Rotate by one of the given angles."""

    def __init__(self, image_size=[300, 300], affine_degrees=5,
                 affine_translate=(0.01, 0.02), affine_scale=(0.9, 1.1)
                 ):
        self.image_size = image_size
        self.affine_degrees = affine_degrees
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale

    def get_transformations(self, train=True):
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


class AlbumentationTransformations:
    def __init__(self, image_size=200, custom_transform=False):
        self.image_size = image_size
        self.custom_transform = custom_transform


    def get_transformations(self, train=True):
        def custom_transformation(image, **kwargs):
            if (self.custom_transform):
                return imp.process_transform(image, transform=self.custom_transform)
            else:
                return image

        if train:
            return Compose([
                #Lambda(image=repeat_ch, name='repeat'),
                Lambda(image=custom_transformation,  name='custom-transform', p=1.),
                Resize(self.image_size, self.image_size, interpolation=cv2.INTER_CUBIC, p=1.),
                # Transpose(p=0.5),
                # HorizontalFlip(p=0.5),
                # VerticalFlip(p=0.5),
                # ShiftScaleRotate(p=0.5),
                # Perspective(p=0.5),
                # ElasticTransform(p=0.5),
                # GridDistortion(p=0.5),
                # CLAHE(p=0.5),
                # Cutout(p=0.25),
                # MotionBlur(p=0.25),
                # ImageCompression(p=0.5, quality_lower=50, quality_upper=100),
                # Affine(scale=[0.5, 1.5], p=0.5),
                # Sharpen(p=0.25),
                # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                # RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                # CoarseDropout(p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                #Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)
        else: 
            return Compose([
                Lambda(image=custom_transformation,  name='custom-transform', p=1.),
                Resize(self.image_size, self.image_size, interpolation=cv2.INTER_CUBIC, p=1.),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                #Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)

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

        true_label = 0 if person_1 == person_2 else 1
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


class BasicDatasetAlbumentation(Dataset):
    def __init__(self, images_folder, compare_file, transform=None, mode=None,
                    test=False
                ):
        self.images_folder = images_folder
        self.transform = transform
        self.mode = mode
        self.test = test

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
        
        if self.test:
            image_1_name = image_1
            image_2_name = image_2
            
        true_label = 0 if person_1 == person_2 else 1
        
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
            image_1 = self.transform(image=image_1)['image']
            image_2 = self.transform(image=image_2)['image']

        if self.test:
            return image_1, image_2, np.array([true_label]), image_1_name, image_2_name
        else:
            return image_1, image_2, np.array([true_label])

    def __len__(self):
        return len(self.pairs)


class BasicStratifiedDatasetAlbumentation(Dataset):
    def __init__(self, images_folder, compare_file, transform=None, mode=None,
                    stratify_each_epoch=False, test=False
                ):
        self.images_folder = images_folder
        self.transform = transform
        self.mode = mode
        self.stratify_each_epoch = stratify_each_epoch
        self.test = test

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
        #print(ix, '/',len(self.pairs))
        # Stratify and shuffle on first batch ()
        if ix == 0 and self.stratify_each_epoch:
            self.pairs = self.stratify_dataset()

        image_1 = self.pairs[ix][0]
        image_2 = self.pairs[ix][1]
        person_1 = image_1.split("_")[0]
        person_2 = image_2.split("_")[0]
        
        if self.test:
            image_1_name = image_1
            image_2_name = image_2
            
        true_label = 0 if person_1 == person_2 else 1
        
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
            image_1 = self.transform(image=image_1)['image']
            image_2 = self.transform(image=image_2)['image']
        if self.test:
            return image_1, image_2, np.array([true_label]), image_1_name, image_2_name
        else:
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


class BasicDatasetTripletRaw(Dataset):
    def __init__(self, images_folder, compare_file, transform=None, mode=None):
        self.transform = transform
        self.mode = mode

        self.compare_file = compare_file

        self.images_folder = images_folder
        self.len_dataset = len(os.listdir(compare_file))

    def __getitem__(self, ix):
        self.triplets = create_triplets_dir(self.compare_file)

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
        return self.len_dataset


class DatasetRawTraining(Dataset):
    def __init__(self, images_folder, compare_file, transform=None, mode=None):
        self.transform = transform
        self.mode = mode

        self.pairs = create_pairs_balanced(compare_file)

        self.images_folder = images_folder

    def __getitem__(self, idx):
        image_1 = self.pairs[idx][0]
        image_2 = self.pairs[idx][1]
        true_label = self.pairs[idx][2]

        image_1 = read(
            os.path.join(self.images_folder, image_1), mode=self.mode
        )
        image_2 = read(
            os.path.join(self.images_folder, image_2), mode=self.mode
        )

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2, np.array([true_label])

    def __len__(self):
        return len(self.pairs)



if __name__ == "__main__":
    # basic_dataset = BasicDatasetTriplet("../data/registred_images_v1_train/",
    #                                     "../compare_files/compare_splited_v1_train_new.txt")

    basic_dataset = BasicDataset("../data/registred_images_v1_train/",
                                 "../compare_files/compare_splited_v1_train_new.txt")

    # basic_dataset = DatasetRawTraining('/mnt/arquivos_linux/1_semestre/Falcao/image_analysis_final_project/images_02/train/',
    #                                    '/mnt/arquivos_linux/1_semestre/Falcao/image_analysis_final_project/images_02/train/')
    #
    print(basic_dataset[0])
