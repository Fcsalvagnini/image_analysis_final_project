import os
import matplotlib.pyplot as plt
import numpy as np


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
