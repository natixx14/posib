"""Main file to launch training of DAE module"""
import logging
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, SiLU, Dropout
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import VisionTransformer

from dae.dataset.constant import CLASSES, TEST_SPLIT
from dae.dataset.loader import GENERATED_HF_DAE, get_teacher_latent
from dae.models.module import Decanted_Student
from dae.trainer.module import Trainer_surrogate
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.utils.prune as prune


class VGG_student(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.features = nn.Sequential(
        Conv2d(3,int(r*64),kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        ReLU(inplace=True),
        Conv2d(int(r*64), int(r*64), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        Conv2d(int(r*64), int(r*128), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(int(r*128), int(r*128), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        Conv2d(int(r*128), int(r*256), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(int(r*256), int(r*256), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(int(r*256), int(r*256), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        Conv2d(int(r*256), int(r*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(int(r*512), int(r*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(int(r*512), int(r*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        Conv2d(int(r*512), int(r*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(int(r*512), int(r*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(int(r*512), int(r*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
                Linear(in_features=int(r*512)*7*7, out_features=4096, bias=True),
                SiLU(),
                Dropout(p=0.5, inplace=False),
                Linear(in_features=4096, out_features=4096, bias=True),
                SiLU(),
                Dropout(p=0.5, inplace=False),
                Linear(in_features=4096, out_features=512, bias=True),
                SiLU(),
                Dropout(p=0.5, inplace=False),
                Linear(in_features=512, out_features=512, bias=True),
                )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

log = logging.getLogger(__name__)

@hydra.main(version_base=None,
            config_path="conf",
            config_name="config")
def launch(cfg: DictConfig) -> None:
    base_cfg = OmegaConf.load(
            os.path.join(
                cfg.dae_model_path,
                '.hydra',
                'config.yaml'
                )
            )
    # SEED
    torch.manual_seed(cfg.random_seed)

    # DEVICE
    device = cfg.device
    if device == "mcuda":
        device = f"cuda:{int(os.environ['SLURM_ARRAY_TASK_ID'])%8}"
    log.info(f"device : {device}")
    
    # DAE MODEL
    log.info("Loading DAE Model")
    log.info(f'Loading BB model {base_cfg.teacher}')
    bb_model, bb_latent, bb_transform = get_teacher_latent(
            base_cfg.teacher,
            base_cfg.dataset,
            base_cfg.teachers_dir,
            cfg.device)
    dae_model = torch.load(
            os.path.join(
                cfg.dae_model_path,
                'dae-final.pk'
                )
            )
    
    # CREATE SAVE DIRECTORY
    path_to_save = './tmp_data'
    os.makedirs(path_to_save, exist_ok=True)
    
    def teacher_forward(instance):
        # PREPROCESSING
        l1_tensor = []
        tensorTr = transforms.ToTensor()
        for i in instance['image']:
            t_i = tensorTr(i)
            if t_i.shape[0] == 1:
                t_i = torch.cat((t_i, t_i, t_i), dim=0)
            t1_i = bb_transform(t_i)
            l1_tensor.append(t1_i.unsqueeze(0))
        t1_img = torch.vstack(l1_tensor).to(cfg.device)
        # INFERENCE
        z = bb_latent(t1_img).view(t1_img.size(0), -1)
        uc = dae_model(z)['uc']
        return {
            'uc': uc.cpu(),
            'tensor': t1_img.cpu()
        }

    if cfg.preload:
        # ITERATE OVER SPLIT
        for split in ['train', TEST_SPLIT[base_cfg.dataset]]:
            os.makedirs(os.path.join(path_to_save, split), exist_ok=True)
            log.info(f'Generating {split} dataset of {base_cfg.dataset}')
            # LOAD DATASET
            dataset = load_from_disk(os.path.join(cfg.dataset_dir,
                                                  base_cfg.dataset))[split]
            dataset = dataset.shuffle()
            test = dataset.map(teacher_forward,
                               batched=True,
                               batch_size=cfg.batch_size,
                               )
            save_file = os.path.join(path_to_save, split)
            test.save_to_disk(os.path.join(path_to_save, split))
            log.info(f'Saved {split} dataset at {save_file}')

 
   # DATASET
    dataset_dir= os.path.join(
            cfg.tensors_dir,
            f"{base_cfg.dataset}/{base_cfg.teacher}_{base_cfg.dataset.split('/')[-1]}"
            )
    log.info(f'Loading data from {dataset_dir}...')
    dataset_train = GENERATED_HF_DAE(data_dir=path_to_save, split='train') 
    dataset_test = GENERATED_HF_DAE(data_dir=path_to_save, split=TEST_SPLIT[base_cfg.dataset])
    dataloader_train = DataLoader(dataset_train, 
            batch_size=cfg.batch_size,
            drop_last=True,
            shuffle=True
            )
    dataloader_test = DataLoader(dataset_test,
            batch_size=cfg.batch_size,
            drop_last=True,
            shuffle=True
            )

    # SURROGATE MODEL
    log.info(f"Loading surrogate Model from teacher {base_cfg.teacher}")
    if base_cfg.teacher=='vgg16':
        latent = VGG_student(r=cfg.pruning_ratio)
    elif base_cfg.teacher=='vitB16':
        latent = VisionTransformer(
            image_size=bb_model.image_size,
            patch_size=bb_model.patch_size,
            num_layers=12,
            num_heads=1,
            hidden_dim=int((cfg.pruning_ratio**.5)*bb_model.hidden_dim),
            mlp_dim=int((cfg.pruning_ratio**.5)*bb_model.mlp_dim),
            dropout=bb_model.dropout,
            attention_dropout=bb_model.attention_dropout,
            num_classes=384,
            representation_size=bb_model.representation_size,
            norm_layer=bb_model.norm_layer
            )
    dae_surr = Decanted_Student(
            teacher_latent=latent,
            encoder_d=dae_model.encoder_c,
            predictor=dae_model.predictor
            )
    dae_surr.to('cuda')
    #dae_surr = torch.nn.parallel.DistributedDataParallel(dae_surr)
    '''
    for child in dae_surr.encoder_d.children():
        try:
            prune.ln_structured(child, 'weight', amount=cfg.pruning_ratio, dim=1, n=float('-inf'))
            log.info(f'pruning {child}')
        except:
            pass
        for m in child.children():
            try:
                prune.ln_structured(m, 'weight', amount=cfg.pruning_ratio, dim=1, n=float('-inf'))
                log.info(f'pruning {m}')
            except:
                pass
            for layer in m.children():
                try:
                    prune.ln_structured(layer, 'weight', amount=cfg.pruning_ratio, dim=1, n=float('-inf'))
                    log.info(f'pruning {layer}')
                except:
                    pass
                for module in layer.children():
                    try:
                        prune.ln_structured(module, 'weight', amount=cfg.pruning_ratio, dim=1, n=float('-inf'))
                        log.info(f'pruning {module}')
                    except:
                        pass
   '''
   # FITTING ROUTINE
    log.info('Loading surrogate trainer')
    trainer = Trainer_surrogate(
            device=device,
            model=dae_surr,
            dae=dae_model,
            teacher=bb_latent,
            r_pred=cfg.r_pred,
            r_uc=cfg.r_uc,
            lr=cfg.lr,
            training_ablation=cfg.training_ablation
            )
    log.info("Starting fitting routine")
    trainer.fit(data=dataloader_train, epochs=cfg.epochs)
    
    # SAVING
    try:
        torch.save(dae_surr, os.path.join(trainer.save_path, "dae-surrogate.pk"))
    except:
        pass
    try:
        torch.save(
            dae_surr.state_dict(),
            os.path.join(trainer.save_path, "dae-surrogate_state.pk"),
        )
    except:
        pass
    log.info(f"Model saved at {trainer.save_path}")
    # PRED PERF EVALUATION
    log.info("Evaluating surrogate predictions")
    perf = trainer.perfo(data=dataloader_test)
    for key, val in perf.items():
        log.info(f"{key}:{val:.3f}")
    # EVALUATING ON ADVERSARIALS
    log.info('Begin evaluation on adversarial samples')

    # LOAD MODEL
    # LOADING ATTACKED DATA
    for eps in [0.005, 0.01, 0.05, 0.1]:
        adv_path = f'{cfg.path_generated_data}/adversarial_fgsm/{base_cfg.dataset}/{base_cfg.teacher}/eps_{eps}/'
        log.info(f'Loading adversarial samples from {adv_path}')
        x_test = np.load(os.path.join(adv_path,'x.npy'))
        y_test = np.load(os.path.join(adv_path,'y.npy'))

        # COMPUTING PERF
        dataset = TensorDataset(
                torch.Tensor(x_test),
                torch.Tensor(y_test)
                ) 
        dataloader = DataLoader(dataset, batch_size=128, drop_last=True)
        acc = 0
        for i, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            l = dae_surr(x)['logits']
            yh = torch.argmax(
                    torch.nn.functional.softmax(l, dim=1),
                    dim=1
                    )
            acc += (yh == y).sum().item() 
        log.info(f"Accuracy on adversarial test examples (eps={eps}): {acc/(len(dataloader)*dataloader.batch_size):.3f}")
    for eps in [0.01, 0.1]:
        adv_path = f'{cfg.path_generated_data}/adversarial/{base_cfg.dataset}/{base_cfg.teacher}/eps_{eps}/'
        log.info(f'Loading adversarial samples from {adv_path}')
        x_test = np.load(os.path.join(adv_path,'x.npy'))
        y_test = np.load(os.path.join(adv_path,'y.npy'))

        # COMPUTING PERF
        dataset = TensorDataset(
                torch.Tensor(x_test),
                torch.Tensor(y_test)
                ) 
        dataloader = DataLoader(dataset, batch_size=128, drop_last=True)
        acc = 0
        for i, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            l = dae_surr(x)['logits']
            yh = torch.argmax(
                    torch.nn.functional.softmax(l, dim=1),
                    dim=1
                    )
            acc += (yh == y).sum().item() 
        log.info(f"Accuracy on adversarial test examples (eps={eps}): {acc/(len(dataloader)*dataloader.batch_size):.3f}")
     
if __name__ == "__main__":
    launch()
