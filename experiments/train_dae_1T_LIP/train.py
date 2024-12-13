"""Main file to launch training of DAE module"""
import logging
import os

import hydra
import numpy as np
import torch
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from dae.dataset.constant import CLASSES, DIM_EMBEDDINGS, TEST_SPLIT
from dae.dataset.loader import GENERATED_HF_DAE, get_teacher_latent, load_head
from dae.models.module import DAE_1T_LIP
from dae.trainer.module import DAE_1T_LIP_Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

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
    # SEED
    torch.manual_seed(cfg.random_seed)

    # DEVICE
    device = cfg.device
    if device == "mcuda":
        device = f"cuda:{int(os.environ['SLURM_ARRAY_TASK_ID'])%8}"
    log.info(f"device : {device}")
   
   # DATASET
    dataset_dir = os.path.join(
        cfg.data_dir, cfg.dataset, 
        f"{cfg.teacher}_{cfg.dataset.split('/')[-1]}")
    log.info(
            f"Loading dataset from existing repository {dataset_dir}"
        )
    dataset_train = GENERATED_HF_DAE(data_dir=dataset_dir, split='train') 
    dataset_test = GENERATED_HF_DAE(data_dir=dataset_dir, split=TEST_SPLIT[cfg.dataset])
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

    # DAE MODEL
    log.info("Loading DAE Model")
    model = DAE_1T_LIP(
            dim_input=DIM_EMBEDDINGS[cfg.teacher],
            dim_ur=cfg.d_latent,
            dim_uc=cfg.d_shared,
            dim_logits=CLASSES[cfg.dataset]
            )

    # TRAINER INIT 
    trainer = DAE_1T_LIP_Trainer(
        device=device,
        model=model,
        lr=cfg.lr,
        training_ablation=cfg.training_ablation,
        r_var_reconstruct=cfg.r_reconstruct,
        r_hsic=cfg.r_hsic,
        r_pred=cfg.r_pred,
        r_lip=cfg.r_lip,
        r_norm_regul=cfg.r_norm_regul
        )

    # FITTING ROUTINE
    log.info("Starting fitting routine")
    trainer.fit(data=dataloader_train, test_data=None, epochs=cfg.epochs)
    
    # SAVING
    torch.save(model, os.path.join(trainer.save_path, "dae-final.pk"))
    torch.save(
        model.state_dict(),
        os.path.join(trainer.save_path, "dae-final_state.pk"),
    )
    log.info(f"Model saved at {trainer.save_path}")
    
    # POST TRAINING PERF
    head = load_head(cfg.dataset, cfg.teacher, device, cfg.teachers_dir).to(device)
    log.info('Computing performance of reconstruction')
    perf = trainer.perfo(data=dataloader_test, head=head)
    for key, val in perf.items():
        log.info(f"{key}:{val:.3f}")
    log.info('Computing performance of ablation')
    abl = trainer.ablation_study(data=dataloader_test, head=head)
    for key, val in abl.items():
        log.info(f"{key}:{val:.3f}")
    
    # EVALUATING ON ADVERSARIALS
    log.info('Begin evaluation on adversarial samples')

    # LOAD MODEL
    log.info(f'Loading BB model {cfg.teacher}')
    bb_model, bb_latent, bb_transform = get_teacher_latent(cfg.teacher, cfg.dataset, cfg.teachers_dir, cfg.device)
    # LOADING ATTACKED DATA
    for eps in [0.01,  0.1, 0.05, 0.005]:
        adv_path = f'{cfg.path_generated_data}/adversarial_fgsm/{cfg.dataset}/{cfg.teacher}/eps_{eps}/'
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
            z = bb_latent(x)
            l = model(z)['y_hat']
            yh = torch.argmax(
                    torch.nn.functional.softmax(l, dim=1),
                    dim=1
                    )
            acc += (yh == y).sum().item() 
        log.info(f"Accuracy on adversarial test attack ({eps}): {acc/(len(dataloader)*dataloader.batch_size):.3f}")


if __name__ == "__main__":
    launch()
