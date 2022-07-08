import os
from constants import *
import torch
import yaml

#with open(f'configs/vit_dataset_balanced.yml', 'r') as file:
#    configs = yaml.load(file, Loader=yaml.FullLoader)


class SaveBestModel:
#vit_model_siamese.pth
    def __init__(self,
      best_valid_loss=float('inf')):
     
        self.best_valid_loss = best_valid_loss
        #self.configs = configs
        

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer, criterion, configs, metric='loss'
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation {metric}: {self.best_valid_loss:.3f}")
            savingName = f'{configs["network"]}_epoch_{epoch}.pth'
            savingPath = os.path.join(configs["path_to_save_model"], savingName)
            torch.save(model.state_dict(), savingPath)
            
            #torch.save({
            #    'epoch': epoch + 1,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'loss': criterion,
            #}, savingPath)
