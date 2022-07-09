import os
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import random


def config_flatten(config, fconfig):
    for key in config:
        if isinstance(config[key], dict):
            fconfig = config_flatten(config[key], fconfig)
        else:
            fconfig[key] = config[key]
    return fconfig


def export_learning_curves(monitoring_metrics, output_folder):
    for metric in monitoring_metrics.keys():
        metrics_on_train = np.array(
            monitoring_metrics[metric]["train"]
        )
        metrics_on_validation = np.array(
            monitoring_metrics[metric]["validation"]
        )
        epochs = np.arange(1, len(metrics_on_train) + 1)

        plt.title(f"{metric.capitalize()} Learning Curves")
        plt.xlabel("Epochs")
        plt.ylabel(f"{metric.capitalize()}")
        plt.plot(epochs, metrics_on_train, 'b', label="train")
        plt.plot(epochs, metrics_on_validation, 'r', label="validation")
        plt.legend()

        plt.tight_layout()
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, f"{metric}.png"), dpi=600)
        plt.close()


def create_triplet(file):
    """
    O(n^2)
    """
    slice_str = 8

    with open(file, "r") as file:
        lines = file.read().splitlines()
        pairs = list(
            map(lambda line: line.split(" "), lines)
        )
        for i in range(len(pairs)):
            if pairs[i][0][:slice_str] == pairs[i][1][:slice_str]:

                position = i
                while pairs[position][0][:slice_str] == pairs[position][1][:slice_str]:
                    position = random.randint(0, len(pairs) - 1)

                pairs[i].append(pairs[position][1])

            else:
                position = i
                while pairs[position][0][:slice_str] == pairs[position][1][:slice_str]:
                    position = random.randint(0, len(pairs) - 1)

                pairs[i].append(pairs[position][1])

        for j in range(len(pairs)):
            if pairs[j][0][:slice_str] == pairs[j][1][:slice_str]:
                continue
            else:
                aux = pairs[j][0]
                pairs[j][0] = pairs[j][1]
                pairs[j][1] = pairs[j][2]
                pairs[j][2] = aux

        triplets = pairs

        return triplets


def create_triplets_dir(compare_dir, id_image=8):
    list_images = os.listdir(compare_dir)

    random.shuffle(list_images)

    triplets = []

    len_dataset_train = len(list_images)

    idx_pos = random.randint(0, len_dataset_train - 1)

    for i in range(len_dataset_train):
        # if diferents, find element
        anchor = list_images[i]

        while list_images[i][:id_image] != list_images[idx_pos][:id_image]:
            idx_pos = random.randint(0, len_dataset_train - 1)

        pos = list_images[idx_pos]

        while list_images[i][:id_image] == list_images[idx_pos][:id_image]:
            idx_pos = random.randint(0, len_dataset_train - 1)

        neg = list_images[idx_pos]

        triplets.append([anchor, pos, neg])

    return triplets


def create_pairs_balanced_2(compare_dir, id_image=8):
    list_label_images = []

    list_images = os.listdir(compare_dir)

    list_images.sort()

    label = 0

    len_dataset = len(list_images)

    for i in range(len(list_images) // 3):
        similars = list_images[i * 3:i * 3 + 3]

        temp = list(combinations(similars, 2))

        for j in range(len(temp)):
            pair = list(temp[j])
            pair += [label]
            list_label_images.append(tuple(pair))

        for j in range(len(temp)):
            pair = list(temp[j])
            pair.reverse()
            pair += [label]
            list_label_images.append(tuple(pair))

    length_dissimilar = len(list_label_images)

    label = 1

    count = 0

    while count < length_dissimilar:

        random.seed(None)

        idx_pos_dissimilar_1 = 0
        idx_pos_dissimilar_2 = 0

        while list_images[idx_pos_dissimilar_1][:id_image] == list_images[idx_pos_dissimilar_2][:id_image]:
            idx_pos_dissimilar_1 = random.randint(0, len_dataset - 1)
            idx_pos_dissimilar_2 = random.randint(0, len_dataset - 1)

        list_label_images.append((list_images[idx_pos_dissimilar_1], list_images[idx_pos_dissimilar_2], label))

        count += 1

    return sorted(list_label_images, key=lambda x: x[1])


def create_pairs_balanced(compare_dir, id_image=8):
    list_label_images = []

    list_images = os.listdir(compare_dir)

    len_dataset_train = len(list_images)

    idx_pos = random.randint(0, len_dataset_train - 1)

    for id_balanced in range(len_dataset_train):
        if id_balanced % 2 == 0:
            label = 0

            while list_images[id_balanced][:id_image] != list_images[idx_pos][:id_image]:
                idx_pos = random.randint(0, len_dataset_train - 1)

            list_label_images.append((list_images[id_balanced], list_images[idx_pos], label))
        else:

            label = 1

            while list_images[id_balanced][:id_image] == list_images[idx_pos][:id_image]:
                idx_pos = random.randint(0, len_dataset_train - 1)

            list_label_images.append((list_images[id_balanced], list_images[idx_pos], label))

    return list_label_images


def create_csv(path,path_image, name):
    pairs = create_pairs_balanced_2(f"{path}{path_image}")

    import pandas as pd

    dict_save_pd = {'image_1': [],
                    'image_2': [],
                    'label': []}

    for elem in pairs:
        dict_save_pd['image_1'].append(elem[0])
        dict_save_pd['image_2'].append(elem[1])
        dict_save_pd['label'].append(elem[2])

    pd.DataFrame(dict_save_pd).to_csv(f"{path}{name}", index=False)

    return True


if __name__ == '__main__':
    PATH = "/mnt/arquivos_linux/1_semestre/Falcao/image_analysis_final_project/image_02_crop/"
    create_csv(PATH, 'validation', 'validation.csv')

    PATH = "/mnt/arquivos_linux/1_semestre/Falcao/image_analysis_final_project/image_02_crop/"
    create_csv(PATH, 'train', 'train.csv')

    PATH = "/mnt/arquivos_linux/1_semestre/Falcao/image_analysis_final_project/image_02_crop/"
    create_csv(PATH, 'test', 'test.csv')

    # for i in range(len(triplas)):
    #     print(triplas[i])
    # create_triplet("../compare_files/compare_splited_v1_test_new.txt")
