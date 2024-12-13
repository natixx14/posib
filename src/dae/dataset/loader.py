import copy
import glob
import logging
import os
import random
from itertools import chain

import numpy as np
import torch
import torchvision.models as m
from dae.dataset.constant import ID_LATENT
from dae.dataset.hbar_models import ResNet18
from datasets import load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights


class IMAGE_DATASET(ImageFolder):
    def __init__(self, root, split, transform=None):
        super().__init__(root=os.path.join(root, split), transform=transform)
    


class IMAGE_PAIR_DATASET(ImageFolder):
    def __init__(self, root, split, transform=None):
        super().__init__(root=os.path.join(root, split), transform=transform)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        dir_path = '/'.join(path.split('/')[:-1])
        list_dir = os.listdir(dir_path)
        path2 = os.path.join(dir_path, random.choice(list_dir))
        sample = self.loader(path)
        sample2 = self.loader(path2)
        if self.transform is not None:
            sample = self.transform(sample)
            sample2 = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, sample2, target

class IMAGE_CONTRASTIVE_DATASET(ImageFolder):
    def __init__(self, root, split, n_pos=10, n_neg=10, transform=None):
        super().__init__(root=os.path.join(root, split), transform=transform)
        self.n_pos = n_pos
    
    def __getitem__(self, index):
        # LOAD DATA
        path, target = self.samples[index]
        class_name = path.split('/')[-1]
        sample = self.loader(path)
        # POSITIVE CLASS
        pos_dir_path = '/'.join(path.split('/')[:-1])
        pos_list_dir = [os.path.join(pos_dir_path,e) for e in os.listdir(pos_dir_path)]
        pos_sample = list(map(self.loader, np.random.choice(pos_list_dir, size=self.n_pos)))
        # NEGATIVE CLASS
        neg_classes = self.classes
        neg_list_dir = [glob.glob(os.path.join('/'.join(path.split('/')[:-2]), neg_path, '*.png')) for neg_path in neg_classes]
        neg_list_dir = list(chain(*neg_list_dir))
        neg_sample = list(map(self.loader, np.random.choice(neg_list_dir, size=self.n_pos)))
        # TRANSFORM
        if self.transform:
            neg_sample = list(map(self.transform, neg_sample))
            pos_sample = list(map(self.transform, pos_sample))
        pos_sample = [t.unsqueeze(0) for t in pos_sample]
        neg_sample = [t.unsqueeze(0) for t in neg_sample]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # FIRE AWAY
        pos_sample = torch.cat(pos_sample, dim=0)
        neg_sample = torch.cat(neg_sample, dim=0)
        return sample, pos_sample, neg_sample, target


class GENERATED_HF_DAE(Dataset):

    def __init__(self,
                data_dir,
                split,
                transform=None):
        # LOGGER
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Loading {split} dataset ...')
        # ATTRIBUTES
        self.transform = transform
        dataset = load_from_disk(os.path.join(data_dir, split))
        self.dataset = dataset.with_format('torch')
    
    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, idx):
        d = self.dataset[idx]
        key_label = 'label'
        key_img = 'image'
        if 'fine_label' in d.keys():
            key_label='fine_label'
        if 'img' in d.keys():
            key_img='img'
        if 'input' in d.keys():
            return d['input'], d[key_label]
        elif 'uc' in d.keys():
            return d['tensor'], d['uc'], d[key_label]
        if 'logits1' in d.keys():
            return d['logits1'], d['latent1'], d['logits2'], d['latent2'], d['label']
        return d['logits'], d['latent'], d[key_label]


def load_black_box(dataset, black_box, device, generated_model_dir=''):
    if black_box=='resnet18':
        model = torch.load(os.path.join(generated_model_dir, dataset, black_box, f'{black_box}.pk'), map_location=device)
        bb_transform = ResNet18_Weights.DEFAULT.transforms()
    elif black_box=='hresnet18' or black_box=='at_hresnet18':
        model = ResNet18()
        model.load_state_dict(
                torch.load(
                    os.path.join(generated_model_dir, dataset, black_box, f'{black_box}.pk'),
                    map_location=device
                    )
                )
        model.rob=True
        model.to(device)
        bb_transform = transforms.Compose([
                        #transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),
            ])
    elif black_box=="vitB16":
        model = torch.load(os.path.join(generated_model_dir, dataset, black_box, f'resnet18.pk'), map_location=device)
        bb_transform = m.ViT_B_16_Weights.DEFAULT.transforms() 
    elif black_box=='vgg16':
        model = torch.load(os.path.join(generated_model_dir, dataset, black_box, f'resnet18.pk'), map_location=device)
        bb_transform = m.VGG16_Weights.DEFAULT.transforms()
    else:
        return
    return model, bb_transform

def load_head(dataset, black_box, device, generated_model_dir=''):
    if black_box=='resnet18':
        model = torch.load(os.path.join(generated_model_dir, dataset, black_box, f'resnet18.pk'), map_location=device)
        head = torch.nn.Sequential(*(list(model.fc.children())[ID_LATENT[black_box]:]))
    elif black_box=='hresnet18':
        model = ResNet18()
        model.load_state_dict(
                torch.load(
                    os.path.join(generated_model_dir, dataset, black_box, f'{black_box}.pk'),
                    map_location=device
                    )
                )
        head = torch.nn.Sequential(model.linear)
    elif black_box=="vitB16":
        model = torch.load(os.path.join(generated_model_dir, dataset, black_box, f'resnet18.pk'), map_location=device)
        head = torch.nn.Sequential(*(list(model.heads.children())[ID_LATENT[black_box]:]))
    elif black_box=='vgg16':
        model = torch.load(os.path.join(generated_model_dir, dataset, black_box, f'resnet18.pk'), map_location=device)
        head = torch.nn.Sequential(*(list(model.classifier.children())[ID_LATENT[black_box]:]))
    else:
        return
    return head

def get_teacher_latent(teacher, dataset, teachers_dir, device='cpu'):
    bb_model, bb_transform = load_black_box(dataset, teacher,
                                            device, teachers_dir)
    bb_latent = copy.deepcopy(bb_model)
    if teacher.split('-')[0] == "vitB16":
        bb_latent.heads = torch.nn.Sequential(*(list(bb_model.heads.children())[:ID_LATENT[teacher]]))
    elif teacher.split('-')[0] == "vgg16":
        bb_latent.classifier = torch.nn.Sequential(*(list(bb_model.classifier.children())[:ID_LATENT[teacher]]))
    elif teacher.split('-')[0] =='resnet18':
        bb_latent.fc = torch.nn.Sequential(*(list(bb_model.fc.children())[:ID_LATENT[teacher]]))
    elif teacher.split('-')[0] =='hresnet18':
        bb_latent.linear = torch.nn.Sequential(
                torch.nn.Identity()
                )
        bb_latent.rob=True
    bb_latent.requires_grad_(False)
    bb_model.requires_grad_(False)
    bb_latent.to(device)
    return bb_model, bb_latent, bb_transform



def main():
    bb_transform = ResNet18_Weights.DEFAULT.transforms()
    d = IMAGE_CONTRASTIVE_DATASET(root='./data/raw/', split='train', transform=bb_transform)
    dataLoader = torch.utils.data.DataLoader(d, batch_size=32, shuffle=True)
    for i, batch in enumerate(dataLoader):
        s, s1, y = batch
        break

if __name__=='__main__':
    main()
