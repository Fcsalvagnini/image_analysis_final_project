from utils import export_learning_curves
from torch_snippets import DataLoader, optim
import torch
from data_loaders import BasicTransformations, BasicDataset
from losses import ContrastiveLoss
from models import SimpleConvSiameseNN
from torchsummary import summary
from tqdm import trange
import gc

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            running_accuracy  += accuracy.cpu()

            progress_bar.set_postfix(
                desc=f"[Epoch {epoch}] Loss: {loss:.3e} - Acc {accuracy:.3e}"
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
                running_accuracy  += accuracy.cpu()

                progress_bar.set_postfix(
                    desc=f"[Epoch {epoch}] Loss: {loss:.3e} - Acc {accuracy:.3e}"
                )

        monitoring_metrics["loss"]["validation"].append(
            running_loss / len(loader)
        )
        monitoring_metrics["accuracy"]["validation"].append(
            running_accuracy / len(loader)
        )

def run_training_experiment(model, train_loader, validation_loader, optimizer, 
        criterion, epochs
    ):
    monitoring_metrics = {
        "loss" : {"train" : [], "validation" : []},
        "accuracy" : {"train" : [], "validation" : []}
    }

    for epoch in range(1, epochs + 1):
        run_train_epoch(
            model, optimizer, criterion, train_loader, monitoring_metrics, epoch
        )
        run_validation(
            model, criterion, validation_loader, monitoring_metrics, epoch
        )

    export_learning_curves(monitoring_metrics, output_folder="../report/dataset_v1/")
    torch.save(model, "../models/first_model.pth")


if __name__ == "__main__":
    transfomation_obj = BasicTransformations(image_size=[120, 120])
    transfomations_composition = transfomation_obj.get_transformations()
    train_dataset = BasicDataset(        
        images_folder = "../data/registred_images_v1_train/",
        compare_file = "../compare_files/compare_splited_v1_train_new.txt",
        transform=transfomations_composition
    )
    validation_dataset = BasicDataset(        
        images_folder = "../data/registred_images_v1_validation/",
        compare_file = "../compare_files/compare_splited_v1_validation_new.txt",
        transform=transfomations_composition
    )
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    epochs = 200
    model = SimpleConvSiameseNN().to(DEVICE)
    summary(model)
    
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(
        model.parameters(), lr = 0.001, weight_decay=0.01
    )

    run_training_experiment(
        model, train_data_loader, validation_data_loader, optimizer, 
        criterion, epochs
    )