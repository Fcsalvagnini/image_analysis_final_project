import argparse
import yaml

import numpy as np
from sklearn.metrics import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_snippets import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


from models import *
from data_loaders import *
from losses import *

FACTORY_DICT = {
    "model": {
        "SimpleConvSiameseNN": SimpleConvSiameseNN,
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
        "BasicStratifiedDatasetAlbumentation": BasicStratifiedDatasetAlbumentation
    },
    "transformation": {
        "BasicTransformations": BasicTransformations,
        "Albumentation": AlbumentationTransformations
    },
    "loss": {
        "ContrastiveLoss": ContrastiveLoss,
        "TripletLoss": TripletLoss,
        "CosineLoss": CosineLoss,
        "ContrastiveCosineLoss": ContrastiveCosineLoss
    }
}

def parse_yaml_file(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations


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


def inference_factory(configs):
    model_configs = configs["model"]  
    criterion_configs = configs["loss"]  
    test_dataset_configs = configs["test_dataset"]

    test_dataset = get_dataset(test_dataset_configs, configs, train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=configs["test"]["batch_size"], shuffle=False
    )

    if type(model_configs) == dict:
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
            **model_configs[list(model_configs.keys())[0]]
        )
    else:
        model = FACTORY_DICT["model"][model_configs]()
    
    criterion = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]](
        **criterion_configs[list(criterion_configs.keys())[0]]
    )

    return model, test_loader, criterion


def load_model_weights(model, model_weights):
    model.load_state_dict(torch.load(model_weights))
    return model


def get_euclidean_distance(features_1, features_2, labels, same, different):
    with torch.no_grad():
        euclidean_distance = F.pairwise_distance(features_1, features_2)
        #print(euclidean_distance.shape)
        if (labels == 0):
            same.append(euclidean_distance.item())
        else:
            different.append(euclidean_distance.item())
    return same, different


def plot_histogram(same, different, configs):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Histograma de distâncias para as duas classes', fontsize = 20, fontweight = 'bold')
    ax.set_xlabel('Distância Euclidiana', fontsize = 16, fontweight = 'bold')
    ax.set_ylabel('Densidade', fontsize = 16, fontweight = 'bold')
    ax.hist(different,bins = 50, alpha = 0.7, label = 'different')
    ax.hist(same, bins = 50, alpha = 0.7, label = 'same')
    ax.tick_params(labelsize = 16, axis = 'both')
    ax.legend()
    ax.grid(True)
    #plt.plot()
    plt.savefig(f'euclidean_distance_histogram_{configs["network"]}_margin_{configs["loss"]["ContrastiveLoss"]["margin"]}_th_{configs["loss"]["ContrastiveLoss"]["contrastive_threshold"]}.png')


def inference(model, test_loader, loss_fn, configs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        accum_loss = 0
        accum_acc = 0
        num_samples = 0
        accum_image_predictions =  []
        accum_target_predictions = []
        same, different = [], []
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for itr, batch in progress_bar:
            image_1, image_2, labels = [data.to(device) for data in batch[:3]]
            image_1_name, image_2_name = batch[3:]

            
            features_1, features_2 = model(image_1, image_2)
            same, different = get_euclidean_distance(
                features_1, features_2, labels, same, different)
            loss, accuracy = loss_fn(features_1, features_2, labels)

            accum_loss += loss.cpu() * configs['test']['batch_size']
            accum_acc +=  accuracy.item() * configs['test']['batch_size']
            num_samples += configs['test']['batch_size']

            if ((itr + 1) % configs['test']['verbose'] == 0) or ((itr + 1) == len(test_loader)):
                description = f'[TEST] acc: {accum_acc / num_samples:.3f} | loss: {accum_loss / num_samples:.3f}'
                progress_bar.set_description(description)
    
    plot_histogram(same, different, configs)
    print(f'[Same] Mean: {np.mean(same):.2f}, std: {np.std(same)}')
    print(f'[Different] {np.mean(different):.2f}, std: {np.std(different)}')


"""
individuo_nada_posDedo_
00002302_00000001_roll_U_01_1.png 00002302_00000001_roll_V_01_1.png


python3 inference.py --config_file configs/sample_config_timm.yml --model_weights ../report/models/efficientnet_b0_epoch_1.pth

"""


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description="Fingerprint Identification Model Training Framework")
    parser.add_argument("--config_file", type=str, help="Path to model weights (.pth)")
    parser.add_argument("--model_weights", type=str, help="Path to model weights (.pth)")
    args = parser.parse_args()
    configurations = parse_yaml_file(args.config_file)
    model_weights = args.model_weights

    model, test_loader, loss_fn = inference_factory(configurations)
    model.to(DEVICE)
    model = load_model_weights(model, model_weights)

    inference(model, test_loader, loss_fn, configurations)
