from torchvision import transforms
from torch_snippets import Dataset, read
import cv2
import os
from skimage.morphology import skeletonize
import numpy as np
import pandas as pd
import random

import images_preprocessing as imp

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Perspective,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Affine,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, RandomResizedCrop,
    Sharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, Lambda, ElasticTransform, ImageCompression, ToFloat, 
)
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.crops.transforms import CenterCrop
from albumentations.augmentations.geometric.rotate import Rotate

from utils import create_triplet, create_pairs_balanced, create_triplets_dir


class BasicTransformations:
    """Rotate by one of the given angles."""

    def __init__(self, image_size=[300, 300], affine_degrees=5,
                 affine_translate=(0.01, 0.02), affine_scale=(0.9, 1.1),
                 rotate_degrees=45, gaussian_blur_kernel=[3, 3],
                 random_erasing_p=0.5, random_erasing_scale=[0.02, 0.33]
                 ):
        self.image_size = image_size
        self.affine_degrees = affine_degrees
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.rotate_degrees = rotate_degrees
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.random_erasing_p = random_erasing_p
        self.random_erasing_scale = random_erasing_scale


    def get_transformations(self, train=True):
        if train:
            transformations_composition = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(self.rotate_degrees),
                transforms.RandomAffine(self.affine_degrees, self.affine_translate,
                                        scale=self.affine_scale),
                transforms.GaussianBlur(self.gaussian_blur_kernel),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.RandomErasing(p=self.random_erasing_p, 
                        scale=self.random_erasing_scale),
                transforms.Normalize((0.5), (0.5))
            ])
        else:
            transformations_composition = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])

        return transformations_composition


class AlbumentationTransformations:
    def __init__(self, image_size=200, custom_transform=False):
        self.image_size = image_size
        self.custom_transform = custom_transform
        self.enhancement = None

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
                #Resize(self.image_size, self.image_size, interpolation=cv2.INTER_CUBIC, p=1.),
                Rotate(limit=90, interpolation=cv2.INTER_LINEAR, p=0.5),
                CenterCrop(self.image_size, self.image_size, p=1.0),
                # Transpose(p=0.5),
                # HorizontalFlip(p=0.5),
                # VerticalFlip(p=0.5),
                # ShiftScaleRotate(p=0.5),
                # Perspective(p=0.5),
                # ElasticTransform(p=0.5),
                # GridDistortion(p=0.5),
                # CLAHE(p=0.5),
                # Cutout(p=0.25),
                # GaussNoise(p=0.5),
                # MedianBlur(p=0.5),
                # MotionBlur(p=0.25),
                # ImageCompression(p=0.5, quality_lower=50, quality_upper=100),
                # Affine(scale=[0.5, 1.5], p=0.5),
                # Sharpen(p=0.25),
                # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                # RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                # CoarseDropout(p=0.5),
                #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)
        else:
            return Compose([
                Lambda(image=custom_transformation,  name='custom-transform', p=1.),
                #Resize(self.image_size, self.image_size, interpolation=cv2.INTER_CUBIC, p=1.),
                CenterCrop(self.image_size, self.image_size, p=1.0),
                #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0, p=1.0),
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


class BasicDatasetCsv(Dataset):
    def __init__(self, images_folder, compare_file, transform=None, mode=None):
        self.transform = transform
        self.mode = mode

        pd_dir_images = pd.read_csv(compare_file)

        self.images_folder = images_folder

        self.images_1 = pd_dir_images['image_1'].tolist()
        self.images_2 = pd_dir_images['image_2'].tolist()
        self.labels = pd_dir_images['label'].tolist()

    def __getitem__(self, ix):
        image_1 = self.images_1[ix]
        image_2 = self.images_2[ix]
        true_label = self.labels[ix]
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
        return len(self.images_1)


class BalancedCroppedDataset(Dataset):
    def __init__(self, images_folder, transform, mode, subset_images, 
                repeat=1, binarize=False, invert=False
            ):
        self.images_folder = images_folder
        self.transform = transform
        self.mode = mode
        self.repeat = repeat
        self.binarize = binarize
        self.invert = invert
        
        with open(subset_images, "r") as file:
            self.cropped_images = file.read().splitlines()


    def get_image_pairs(self):
        pairs = []
        subject_ids = [image.split("_")[0] for image in self.cropped_images] 
        subject_ids = np.unique(subject_ids)

        for subject_id in subject_ids:
            subject_images = [
                img for img in self.cropped_images \
                if subject_id in img
            ]
            other_subject_images = [
                img for img in self.cropped_images \
                if subject_id not in img
            ]
            
            for subject_image_1 in subject_images:
                # Adds similar pairs
                for subject_image_2 in subject_images:
                    if subject_image_1 == subject_image_2:
                        continue
                    pairs.append([
                        subject_image_1, subject_image_2
                    ])
                # Adds dissimilar pairs
                for other_subject_image in random.sample(other_subject_images, 2):
                    # In case of randomly selecting the same image
                    while other_subject_image == pairs[-1][1]:
                        other_subject_image = random.sample(other_subject_images, 1)
                    pairs.append([
                        subject_image_1, other_subject_image
                    ])

        np.random.shuffle(pairs)

        return pairs

    def __getitem__(self, ix):
        if ix == 0:
            self.pairs = []
            for _ in range(self.repeat):
                self.pairs += self.get_image_pairs()

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

        if self.binarize:
            image_1 = cv2.adaptiveThreshold(
                image_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C
                27, 13)
            image_2 = cv2.adaptiveThreshold(
                image_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C
                27, 13)

        if self.invert:
            image_1 = 255 - image_1
            image_2 = 255 - image_2

        if not self.mode:
            image_1 = np.expand_dims(image_1, 2)
            image_2 = np.expand_dims(image_2, 2)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2, np.array([true_label])

    def __len__(self):
        return len(self.cropped_images) * 4 * self.repeat


class BalancedCroppedDatasetAlbumentation(Dataset):
    def __init__(self, images_folder, transform, mode, subset_images, repeat=1):
        self.images_folder = images_folder
        self.transform = transform
        self.mode = mode
        self.repeat = repeat
        
        with open(subset_images, "r") as file:
            self.cropped_images = file.read().splitlines()


    def get_image_pairs(self):
        pairs = []
        subject_ids = [image.split("_")[0] for image in self.cropped_images] 
        subject_ids = np.unique(subject_ids)

        for subject_id in subject_ids:
            subject_images = [
                img for img in self.cropped_images \
                if subject_id in img
            ]
            other_subject_images = [
                img for img in self.cropped_images \
                if subject_id not in img
            ]
            
            for subject_image_1 in subject_images:
                # Adds similar pairs
                for subject_image_2 in subject_images:
                    if subject_image_1 == subject_image_2:
                        continue
                    pairs.append([
                        subject_image_1, subject_image_2
                    ])
                # Adds dissimilar pairs
                for other_subject_image in random.sample(other_subject_images, 2):
                    # In case of randomly selecting the same image
                    while other_subject_image == pairs[-1][1]:
                        other_subject_image = random.sample(other_subject_images, 1)
                    pairs.append([
                        subject_image_1, other_subject_image
                    ])

        np.random.shuffle(pairs)

        return pairs

    def __getitem__(self, ix):
        if ix == 0:
            self.pairs = []
            for _ in range(self.repeat):
                self.pairs += self.get_image_pairs()

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
            image_1 = self.transform(image=image_1)['image']
            image_2 = self.transform(image=image_2)['image']

        return image_1, image_2, np.array([true_label])

    def __len__(self):
        return len(self.cropped_images) * 4 * self.repeat


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
        # print(ix, '/',len(self.pairs))
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
