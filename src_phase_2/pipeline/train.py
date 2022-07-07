import os
import sys
import random

import argparse
import yaml
import json

import cv2
import numpy as np
from tqdm import tqdm

from numpy.random.mtrand import RandomState
from sklearn.model_selection import KFold

# import apex
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Perspective,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Affine,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, RandomResizedCrop,
    Sharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, ElasticTransform, ImageCompression, ToFloat
)
from albumentations.pytorch import ToTensorV2

from utils import *
from FPSim import *
from dataset import FingerprintDataset
from losses import *


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_train_transforms():
    return Compose([
        Resize(config['training']['image_size'], config['training']['image_size'], interpolation=cv2.INTER_CUBIC, p=1.),
        # Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        # ShiftScaleRotate(p=0.5),
        # Perspective(p=0.5),
        # ElasticTransform(p=0.5),
        # GridDistortion(p=0.5),
        # CLAHE(p=0.5),
        # Cutout(p=0.25),
        # MotionBlur(p=0.25),
        # ImageCompression(p=0.5, quality_lower=50, quality_upper=100),
        # Affine(scale=[0.5, 1.5], p=0.5),
        Sharpen(p=0.25),
        # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        # RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        # CoarseDropout(p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms():
    return Compose([
        Resize(config['training']['image_size'], config['training']['image_size'], interpolation=cv2.INTER_CUBIC, p=1.),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def train_epoch(epoch, model, loss_function, scheduler, optimizer, train_loader, scaler, device):
    model.train()

    accum_loss = 0
    accum_acc = 0
    num_samples = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for itr, (image1, image2, label) in progress_bar:
        images1 = image1.to(device)  # , dtype=torch.float32)
        images2 = image2.to(device)  # , dtype=torch.float32
        labels = label.to(device)

        with autocast():
            codesA, codesB = model(images1, images2)
            loss, acc = loss_function(codesA, codesB, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        samples = labels.shape[0]
        num_samples += samples
        accum_loss += loss.item() * samples
        accum_acc += acc.item() * samples

        # scaler.scale(loss).backward()
        #
        # if ((itr + 1) % config['training']['accum_iter'] == 0) or ((itr + 1) == len(train_loader)):
        #    scaler.step(optimizer)
        #    scale = scaler.get_scale()
        #    scaler.update()
        #    skip_scheduler = (scale != scaler.get_scale())
        #    optimizer.zero_grad()
        #
        #    if scheduler is not None and not skip_scheduler:
        #        scheduler.step()

        if ((itr + 1) % config['training']['verbose_step'] == 0) or ((itr + 1) == len(train_loader)):
            description = f'[TRAIN] epoch {epoch} loss: {accum_loss / num_samples:.4f} | acc: {accum_acc / num_samples:.4f}'
            progress_bar.set_description(description)

    return accum_acc / num_samples, accum_loss / num_samples


def valid_epoch(epoch, model, loss_function, valid_loader, device):
    model.eval()

    accum_loss = 0
    accum_acc = 0
    num_samples = 0

    progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for itr, (image1, image2, label) in progress_bar:
        images1 = image1.to(device)
        images2 = image2.to(device)
        labels = label.to(device)

        codesA, codesB = model(images1, images2)
        loss, acc = loss_function(codesA, codesB, labels)

        samples = labels.shape[0]
        num_samples += samples
        accum_loss += loss.item() * samples
        accum_acc += acc.item() * samples

        if ((itr + 1) % config['training']['verbose_step'] == 0) or ((itr + 1) == len(valid_loader)):
            description = f'[VALID] epoch {epoch} loss: {accum_loss / num_samples:.4f} | acc: {accum_acc / num_samples:.4f}'
            progress_bar.set_description(description)
    return accum_acc / num_samples, accum_loss / num_samples


def config_flatten(config, fconfig):
    for key in config:
        if isinstance(config[key], dict):
            fconfig = config_flatten(config[key], fconfig)
        else:
            fconfig[key] = config[key]
    return fconfig


def config_log(config, keys):
    log = {}
    for key in keys:
        log[key] = config[key]
    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    configFile = args.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    rng = RandomState()
    if configFile.split('.')[-1] == 'json':
        config = read_config_json(configFile)
    elif configFile.split('.')[-1] == 'yaml':
        config = read_config_yaml(configFile)
    # print(config)
    fconfig = {}
    fconfig = config_flatten(config, fconfig)
    _config_log = config_log(fconfig, ['network', 'train_batch', 'valid_batch', 'image_size',
                                       'epochs', 'lr', 'scheduler', 'margin', 'contrastive_thresh',
                                       'data_path'])
    if config['wandb']:
        import wandb

        wandb.init(project="fp-test",
                   reinit=True,
                   config=fconfig,
                   notes="Testing wandb implementation")
        # wandb.log(_config_log)

    pairs_df = read_txt_dataframe(config['data_nlets'])

    kf = KFold(n_splits=config['training']['num_splits'])
    split = kf.split(pairs_df)

    for fold, (train_idx, valid_idx) in enumerate(split):
        train_df = pairs_df.iloc[train_idx].reset_index(drop=True)
        valid_df = pairs_df.iloc[valid_idx].reset_index(drop=True)

        train_dataset = FingerprintDataset(train_df,
                                           config['data_path'],
                                           get_train_transforms())
        valid_dataset = FingerprintDataset(valid_df,
                                           config['data_path'],
                                           get_valid_transforms())

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['dataloader']['train_batch'],
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=config['dataloader']['num_workers'],
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config['dataloader']['valid_batch'],
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=config['dataloader']['num_workers'],
        )

        model = SiameseNetworkTimmBackbone(config['network'],
                                           config['training']['image_size'],
                                           config['training']['nchannels'],
                                           transformers=False)
        model.to(device)

        scaler = GradScaler()
        # optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=config['training']['lr'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

        if config['training']['scheduler'] == 'stepLr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1,
                                                        step_size=config['training']['epochs'] - 1)
        if config['training']['scheduler'] == "cosineAnnealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['T_0'], T_mult=1,
                                                                             eta_min=config['min_lr'], last_epoch=-1)
        if config['training']['scheduler'] == "oneCycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25,
                                                            max_lr=config['training']['max_lr'],
                                                            epochs=config['training']['epochs'],
                                                            steps_per_epoch=len(train_loader))
        loss = ContrastiveLoss(margin=config['training']['loss']['margin'],
                               contrastive_thresh=config['training']['loss']['contrastive_thresh'])
        if config['wandb']:
            wandb.watch(model, loss, log="all", log_freq=1)
        for epoch in range(config['training']['epochs']):
            train_acc, train_loss = train_epoch(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                loss_function=loss,
                scheduler=scheduler,
                scaler=scaler,
                device=device
            )

            with torch.no_grad():
                valid_acc, valid_loss = valid_epoch(
                    epoch=epoch,
                    model=model,
                    loss_function=loss,
                    valid_loader=valid_loader,
                    device=device
                )
            if config['wandb']:
                wandb.log({'train_acc': train_acc, 'train_loss': train_loss,
                           'valid_acc': valid_acc, 'valid_loss': valid_loss})
