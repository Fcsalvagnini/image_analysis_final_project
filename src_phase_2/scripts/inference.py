import numpy as np

import torch
import torch.nn as nn
#from timm.utils import metrics
from tqdm import tqdm


def inference(model, test_loader, loss_fn, configs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        accum_loss = 0
        accum_acc = 0
        num_samples = 0
        accum_image_predictions =  []
        accum_target_predictions = []

        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for itr, batch in progress_bar:
            image_1, image_2, labels = [data.to(device) for data in batch[:3]]
            image_1_name, image_2_name = batch[3:]

            features_1, features_2 = model(image_1, image_2)
            loss, accuracy = loss_fn(features_1, features_2, labels)

            accum_loss += loss.cpu() * configs['test']['batch_size']
            accum_acc +=  accuracy.item() * configs['test']['batch_size']
            num_samples += configs['test']['batch_size']

            if ((itr + 1) % configs['test']['verbose'] == 0) or ((itr + 1) == len(test_loader)):
                description = f'[TEST] acc: {accum_acc / num_samples:.3f} | loss: {accum_loss / num_samples:.3f}'
                progress_bar.set_description(description)
