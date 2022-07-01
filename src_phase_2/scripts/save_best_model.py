from constants import *
import torch
import yaml

with open(f'{ABS_PATH}/scripts/configs/vit_dataset_balanced.yml', 'r') as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)


class SaveBestModel:

    def __init__(
            self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, configs["path_to_save_model"])
