import os
from torch_snippets import DataLoader, optim
from torch import optim
import torch
from constants import *
from tqdm import trange
import gc
import yaml
import wandb
import argparse
from torchsummary import summary
import os

from utils import export_learning_curves, config_flatten
from data_loaders import BasicDataset, BasicStratifiedDataset, \
    BasicDatasetTriplet, BasicDatasetTripletRaw, BasicTransformations,BasicDatasetAlbumentation, \
    DatasetRawTraining, BasicStratifiedDatasetAlbumentation, AlbumentationTransformations, BasicDatasetCsv, \
    BalancedCroppedDataset, BalancedCroppedDatasetAlbumentation
from inference import inference
from save_best_model import SaveBestModel
from losses import ContrastiveLoss, TripletLoss, CosineLoss, ContrastiveCosineLoss
from models import SimpleConvSiameseNN, PreTrainedVGGSiameseNN, ViTSiamese, \
                    ViTSiameseTriplet, SiameseNetworkTimmBackbone, \
                    ConvFingerprintSiamese

save_best_model = SaveBestModel()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FACTORY_DICT = {
    "model": {
        "SimpleConvSiameseNN": SimpleConvSiameseNN,
        "ConvFingerprintSiamese": ConvFingerprintSiamese,
        "PreTrainedVGGSiameseNN": PreTrainedVGGSiameseNN,
        "ViTSiamese": ViTSiamese,
        "ViTSiameseTriplet": ViTSiameseTriplet,
        "SiameseNetworkTimmBackbone": SiameseNetworkTimmBackbone
    },
    "dataset": {
        "BasicDataset": BasicDataset,
        "BasicStratifiedDataset": BasicStratifiedDataset,
        "BasicDatasetTriplet": BasicDatasetTriplet,
        "DatasetRawTraining": DatasetRawTraining,
        "BasicDatasetTripletRaw": BasicDatasetTripletRaw,
        "BasicDatasetAlbumentation": BasicDatasetAlbumentation,
        "BasicStratifiedDatasetAlbumentation": BasicStratifiedDatasetAlbumentation,
        "BasicDatasetCsv": BasicDatasetCsv,
        "BalancedCroppedDataset": BalancedCroppedDataset,
        "BalancedCroppedDatasetAlbumentation": BalancedCroppedDatasetAlbumentation
    },
    "transformation": {
        "BasicTransformations": BasicTransformations,
        "Albumentation": AlbumentationTransformations
    },
    "optimizer": {
        "Adam": optim.Adam
    },
    "loss": {
        "ContrastiveLoss": ContrastiveLoss,
        "TripletLoss": TripletLoss,
        "CosineLoss": CosineLoss,
        "ContrastiveCosineLoss": ContrastiveCosineLoss
    }
}


def get_dataset(dataset_configs, configs, train=True):
    transform_type = list(dataset_configs.values())[0]["transform"]
    if transform_type:
        transformations_configs = configs["transformation"]
        transfomations_obj = FACTORY_DICT["transformation"][transform_type](
            **transformations_configs[transform_type]
        )
        list(dataset_configs.values())[0]["transform"] = transfomations_obj.get_transformations(train)

    dataset = FACTORY_DICT["dataset"][list(dataset_configs.keys())[0]](
        **dataset_configs[list(dataset_configs.keys())[0]]
    )

    return dataset


def experiment_factory(configs):
    train_dataset_configs = configs["train_dataset"]
    validation_dataset_configs = configs["validation_dataset"]
    test_dataset_configs = configs["test_dataset"]
    model_configs = configs["model"]
    optimizer_configs = configs["optimizer"]
    criterion_configs = configs["loss"]

    # Construct the dataloaders with any given transformations (if any)
    train_dataset = get_dataset(train_dataset_configs, configs)
    validation_dataset = get_dataset(validation_dataset_configs, configs, train=False)
    test_dataset = get_dataset(test_dataset_configs, configs, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=False
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=configs["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=configs["test"]["batch_size"], shuffle=False
    )

    # Build model
    if type(model_configs) == dict:
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
            **model_configs[list(model_configs.keys())[0]]
        )
    else:
        model = FACTORY_DICT["model"][model_configs]()

    optimizer = FACTORY_DICT["optimizer"][list(optimizer_configs.keys())[0]](
        model.parameters(), **optimizer_configs[list(optimizer_configs.keys())[0]]
    )
    criterion = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]](
        **criterion_configs[list(criterion_configs.keys())[0]]
    )

    return model, train_loader, validation_loader, test_loader, optimizer, criterion


def parse_yaml_file(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations


def run_train_epoch(model, optimizer, criterion, loader,
                    monitoring_metrics, epoch, batch_size):
    model.train()
    model.to(DEVICE)
    running_loss = 0
    running_accuracy = 0
    with trange(len(loader), desc='Train Loop') as progress_bar:
        for batch_idx, batch in zip(progress_bar, loader):
            # if        
            image_1, image_2, labels = [data.to(DEVICE) for data in batch]
            optimizer.zero_grad()
            features_1, features_2 = model(image_1, image_2)
            loss, accuracy = criterion(features_1, features_2, labels)
            # image_anchor, image_pos, image_neg = [data.to(DEVICE) for data in batch]
            # optimizer.zero_grad()
            # image_anchor, image_pos, image_neg = model(image_anchor, image_pos, image_neg)
            # loss, accuracy = criterion(image_anchor, image_pos, image_neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu()
            running_accuracy += accuracy.cpu()

            progress_bar.set_postfix(
                desc=f"[Epoch {epoch}] Loss: {running_loss / (batch_idx + 1):.3f} - Acc {running_accuracy / (batch_idx + 1):.3f}"
            )
    epoch_loss = (running_loss / len(loader)).detach().numpy()
    epoch_acc = (running_accuracy / len(loader)).detach().numpy()
    monitoring_metrics["loss"]["train"].append(epoch_loss)
    monitoring_metrics["accuracy"]["train"].append(epoch_acc)

    return epoch_loss, epoch_acc


def run_validation(model, optimizer, criterion, loader, monitoring_metrics,
                   epoch, batch_size):
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to(DEVICE)
        model.eval()
        running_loss = 0
        running_accuracy = 0
        with trange(len(loader), desc='Validation Loop') as progress_bar:
            for batch_idx, batch in zip(progress_bar, loader):
                image_1, image_2, labels = [data.to(DEVICE) for data in batch]
                features_1, features_2 = model(image_1, image_2)
                loss, accuracy = criterion(features_1, features_2, labels)

                running_loss += loss.cpu()
                running_accuracy += accuracy.cpu()

                progress_bar.set_postfix(
                    desc=f"[Epoch {epoch}] Loss: {running_loss / (batch_idx + 1):.3f} - Acc {running_accuracy / (batch_idx + 1):.3f}"
                )

    epoch_loss = (running_loss / len(loader)).detach().numpy()
    epoch_acc = (running_accuracy / len(loader)).detach().numpy()

    monitoring_metrics["loss"]["validation"].append(epoch_loss)
    monitoring_metrics["accuracy"]["validation"].append(epoch_acc)
    save_best_model(epoch_loss,
                epoch,
                model,
                optimizer,
                criterion,
                configurations,
                metric='loss')

    return epoch_loss, epoch_acc


def run_training_experiment(model, train_loader, validation_loader, optimizer,
                            criterion, configs
                            ):
    os.makedirs(configs["path_to_save_model"], exist_ok=True)
    monitoring_metrics = {
        "loss": {"train": [], "validation": []},
        "accuracy": {"train": [], "validation": []}
    }
    # if configs['wandb']:
    #     wandb.watch(model, criterion, log="all", log_freq=1)
    for epoch in range(1, configs["epochs"] + 1):
        train_loss, train_acc = run_train_epoch(
            model, optimizer, criterion, train_loader, monitoring_metrics,
            epoch, batch_size=configs["batch_size"]
        )
        valid_loss, valid_acc = run_validation(
            model, optimizer, criterion, validation_loader, monitoring_metrics,
            epoch, batch_size=configs["batch_size"]
        )
        if configs['wandb']:
            wandb.log({'train_acc': train_acc, 'train_loss': train_loss,
                       'valid_acc': valid_acc, 'valid_loss': valid_loss})

    savingName = f'{configs["network"]}_epoch_{epoch}.pth'
    savingPath = os.path.join(configs["path_to_save_model"], savingName)
    torch.save(model.state_dict(), savingPath)
    export_learning_curves(monitoring_metrics, output_folder=configs["path_to_save_report"])

if __name__ == "__main__":
    # Reads YAML that sets configurations for training experiment
    parser = argparse.ArgumentParser(description="Fingerprint Identification Model Training Framework")
    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    configurations = parse_yaml_file(args.config_file)

    model, train_loader, validation_loader, test_loader, \
        optimizer, criterion = experiment_factory(configurations)

    #summary(model)
    fconfigurations = {}
    fconfigurations = config_flatten(configurations, fconfigurations)
    if configurations['wandb']:
        wandb.init(project="fp-scripts",
                   reinit=True,
                   config=fconfigurations,
                   notes="Testing wandb implementation")

    run_training_experiment(
        model, train_loader, validation_loader, optimizer,
        criterion, configurations
    )

    test_metrics = inference(model, test_loader, criterion, configurations)

    if configurations['wandb']:
        wandb.log(test_metrics)

    torch.cuda.empty_cache()