import torch
import apex
import torch.nn as nn
from torch.optim import lr_scheduler

class Scheduler(nn.Module):
    def __init__(self, optimizer, fn_name, **kwargs):
        super().__init__()
        scheduler = getattr(lr_scheduler, fn_name)(**kwargs)


#class 
