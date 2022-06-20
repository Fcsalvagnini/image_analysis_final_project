from utils import export_learning_curves
from torch_snippets import DataLoader, optim
import torch
from data_loaders import BasicTransformations, BasicDataset
from losses import ContrastiveLoss
from models import SimpleConvSiameseNN, PreTrainedVGGSiameseNN, ViTSiamese
from torchsummary import summary
from tqdm import trange
import gc
import yaml
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FACTORY_DICT = {
    "model": {
        "SimpleConvSiameseNN": SimpleConvSiameseNN,
        "PreTrainedVGGSiameseNN": PreTrainedVGGSiameseNN,
        "ViTSiamese": ViTSiamese
    },
    "dataset": {
        "BasicDataset": BasicDataset
    },
    "transformation": {
        "BasicTransformations": BasicTransformations
    },
    "optimizer": {
        "Adam": optim.Adam
    },
    "loss": {
        "ContrastiveLoss": ContrastiveLoss
    }
}


def get_dataset(dataset_configs, configs):
    transform_type = list(dataset_configs.values())[0]["transform"]
    if transform_type:
        transformations_configs = configs["transformation"]
        transfomations_obj = FACTORY_DICT["transformation"][transform_type](
            **transformations_configs[transform_type]
        )
        list(dataset_configs.values())[0]["transform"] = transfomations_obj.get_transformations()

    dataset = FACTORY_DICT["dataset"][list(dataset_configs.keys())[0]](
        **dataset_configs[list(dataset_configs.keys())[0]]
    )

    return dataset


def experiment_factory(configs):
    train_dataset_configs = configs["train_dataset"]
    validation_dataset_configs = configs["validation_dataset"]
    model_configs = configs["model"]
    optimizer_configs = configs["optimizer"]
    criterion_configs = configs["loss"]

    # Construct the dataloaders with any given transformations (if any)
    train_dataset = get_dataset(train_dataset_configs, configs)
    validation_dataset = get_dataset(validation_dataset_configs, configs)
    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=configs["batch_size"], shuffle=True
    )

    model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
        **model_configs[list(model_configs.keys())[0]]
    )
    optimizer = FACTORY_DICT["optimizer"][list(optimizer_configs.keys())[0]](
        model.parameters(), **optimizer_configs[list(optimizer_configs.keys())[0]]
    )
    criterion = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]](
        **criterion_configs[list(criterion_configs.keys())[0]]
    )

    return model, train_loader, validation_loader, optimizer, criterion


def parse_yaml_file(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations


def run_train_epoch(model, optimizer, criterion, loader, monitoring_metrics, epoch):
    model.train()
    model.to(DEVICE)
    running_loss = 0
    running_accuracy = 0
    with trange(len(loader), desc='Train Loop') as progress_bar:
        for _, batch in zip(progress_bar, loader):
            image_1, image_2, labels = [data.to(DEVICE) for data in batch]
            optimizer.zero_grad()
            features_1, features_2 = model(image_1, image_2)
            loss, accuracy = criterion(features_1, features_2, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu()
            running_accuracy += accuracy.cpu()

            progress_bar.set_postfix(
                desc=f"[Epoch {epoch}] Loss: {running_loss:.3e} - Acc {running_accuracy:.3e}"
            )

    monitoring_metrics["loss"]["train"].append(
        (running_loss / len(loader)).detach().numpy()
    )
    monitoring_metrics["accuracy"]["train"].append(
        (running_accuracy / len(loader)).detach().numpy()
    )


def run_validation(model, criterion, loader, monitoring_metrics, epoch):
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to(DEVICE)
        model.eval()
        running_loss = 0
        running_accuracy = 0
        with trange(len(loader), desc='Validation Loop') as progress_bar:
            for _, batch in zip(progress_bar, loader):
                image_1, image_2, labels = [data.to(DEVICE) for data in batch]
                features_1, features_2 = model(image_1, image_2)
                loss, accuracy = criterion(features_1, features_2, labels)

                running_loss += loss.cpu()
                running_accuracy += accuracy.cpu()

                progress_bar.set_postfix(
                    desc=f"[Epoch {epoch}] Loss: {running_loss:.3e} - Acc {running_accuracy:.3e}"
                )

        monitoring_metrics["loss"]["validation"].append(
            running_loss / len(loader)
        )
        monitoring_metrics["accuracy"]["validation"].append(
            running_accuracy / len(loader)
        )


def run_training_experiment(model, train_loader, validation_loader, optimizer,
                            criterion, configs
                            ):
    monitoring_metrics = {
        "loss": {"train": [], "validation": []},
        "accuracy": {"train": [], "validation": []}
    }

    for epoch in range(1, configs["epochs"] + 1):
        run_train_epoch(
            model, optimizer, criterion, train_loader, monitoring_metrics, epoch
        )
        run_validation(
            model, criterion, validation_loader, monitoring_metrics, epoch
        )

    export_learning_curves(monitoring_metrics, output_folder=configs["path_to_save_report"])
    torch.save(model, configs["path_to_save_model"])


if __name__ == "__main__":
    # Reads YAML that sets configurations for training experiment
    parser = argparse.ArgumentParser(description="Fingerprint Identification Model Training Framework")
    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    configurations = parse_yaml_file(args.config_file)

    model, train_loader, validation_loader, optimizer, criterion = experiment_factory(configurations)

    summary(model)

    run_training_experiment(
        model, train_loader, validation_loader, optimizer,
        criterion, configurations
    )