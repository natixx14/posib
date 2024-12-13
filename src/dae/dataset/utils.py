import os

import torch
from torchvision.models import ResNet18_Weights
from torchvision.transforms import ToTensor, Resize, Compose


def load_pre_trained(path, model=None, device="cpu"):
    m = None
    transform = None
    if model=='resnet18':
        m = torch.load(os.path.join(path, f'{model}.pk'), map_location=device)
        transform = torch.load(os.path.join(path, f'transform.pk'))
    else:
        transform = Compose([Resize(64), ToTensor()])
    return m, transform
