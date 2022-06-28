import os
import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    triplas = create_triplet("../compare_files/compare_splited_v1_train_new.txt")
    for i in range(5, 20):
        print(triplas[i])
    # create_triplet("../compare_files/compare_splited_v1_test_new.txt")
