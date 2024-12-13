from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from dae.models.resnet18 import ResNet18Dec, ResNet18Enc
import logging

log = logging.getLogger(__name__)

class Decanted_Student_fp(nn.Module):
    
    def __init__(self, teacher_latent, encoder_d, predictor):
        super().__init__()
        self.teacher_latent=teacher_latent
        self.teacher_latent.requires_grad_(True)
        log.info(f'teacher latent: {sum(p.numel() for p in self.teacher_latent.parameters() if p.requires_grad)} parameters')
        self.encoder_d=encoder_d
        log.info(f'decanted encoder: {sum(p.numel() for p in self.encoder_d.parameters() if p.requires_grad)} parameters')
        self.predictor=predictor
        log.info(f'predictor head: {sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)} parameters')
        log.info(f'dae surr: {sum(p.numel() for p in self.parameters() if p.requires_grad)} parameters')



    def forward(self, x):
        z = self.teacher_latent(x)#.view(x.size(0), -1)
        uc = self.encoder_d(z)
        logits = self.predictor(uc)
        return logits

class Decanted_Student(nn.Module):
    
    def __init__(self, teacher_latent, encoder_d, predictor):
        super().__init__()
        self.teacher_latent=teacher_latent
        self.teacher_latent.requires_grad_(True)
        log.info(f'teacher latent: {sum(p.numel() for p in self.teacher_latent.parameters() if p.requires_grad)} parameters')
        self.encoder_d=encoder_d
        log.info(f'decanted encoder: {sum(p.numel() for p in self.encoder_d.parameters() if p.requires_grad)} parameters')
        self.predictor=predictor
        log.info(f'predictor head: {sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)} parameters')
        log.info(f'dae surr: {sum(p.numel() for p in self.parameters() if p.requires_grad)} parameters')



    def forward(self, x):
        z = self.teacher_latent(x)#.view(x.size(0), -1)
        uc = self.encoder_d(z)
        logits = self.predictor(uc)
        return {
                'uc': uc,
                'logits': logits
                }

@dataclass(unsafe_hash=True)
class DAE_SURROGATE(nn.Module):
    dim_uc: int= 7
    dim_y: int = 10

    def __post_init__(self):
        super().__init__()
        # ACTIVATION
        self.relu = nn.ReLU()
        # CNN ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                                stride=(2, 2),
                                return_indices=False),
            nn.Conv2d(in_channels=5,
                            out_channels=5,
                            kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                                stride=(2, 2),
                                return_indices=False)
            )
        # MLP ENCODER
        self.fl = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features=14045,
                             out_features=500),  # 188150 1043750
            nn.ReLU(),
            nn.Linear(in_features=500,
                             out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10,
                             out_features=self.dim_uc),
            )
        self.predictor = nn.Sequential(
                nn.Linear(in_features=self.dim_uc,
                             out_features=self.dim_uc*10),
                nn.SiLU(),
                nn.Linear(in_features=self.dim_uc*10,
                             out_features=self.dim_y)
                )
    
    def forward(self, x):
        z = self.encoder(x)
        z_fl = self.fl(z)
        uc = self.fc(z_fl)
        y = self.predictor(uc)
        return {
                'uc': uc,
                'logits': y
                }

@dataclass(unsafe_hash=True)
class DAE_1T_LIP(nn.Module):
    dim_uc: int = 10
    dim_ur: int = 10
    dim_input: int = 32
    dim_logits: int = 10

    def __post_init__(self):
        super().__init__()
        self.dim_shared=self.dim_uc
        self.encoder_r = nn.Sequential(
            nn.Linear(self.dim_input,
                      (self.dim_input) // 3),
            nn.SiLU(),
            nn.Linear(
                (self.dim_input) // 3,
                 self.dim_ur),
        )
        self.encoder_c = nn.Sequential(
            nn.Linear(self.dim_input,
                      (self.dim_input) // 1),
            nn.SiLU(),
            nn.Linear(self.dim_input, self.dim_input*30),
            nn.SiLU(),
            nn.Linear(self.dim_input*30, self.dim_input*10),
            nn.SiLU(),
            nn.Linear(self.dim_input*10, self.dim_input),
            nn.SiLU(),
            nn.Linear(self.dim_input, self.dim_input),
            nn.SiLU(),
            nn.Linear(
                (self.dim_input) // 1,
                self.dim_uc),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.dim_uc + self.dim_ur,
                      self.dim_uc + self.dim_ur),
            nn.SiLU(),
            nn.Linear(self.dim_uc + self.dim_ur,
                      self.dim_input)
            )
        self.predictor = nn.Sequential(
                # nn.Linear(self.dim_uc, self.dim_uc // 4),
                # nn.SiLU(),
                nn.Linear(self.dim_uc , self.dim_logits)
                )

    def forward(self, z, device="cuda"):
        # latent = self.encoder(z)
        # SPLIT LATENT TO GET VARIABLES
        # ur = latent[:, 0:self.dim_ur]
        # uc = latent[:,
        #                   self.dim_ur:self.dim_ur +
        #                   self.dim_uc]
        ur = self.encoder_r(z)
        uc = self.encoder_c(z)
        latent = torch.hstack((ur, uc)).to(device) 
        # DECODE LATENT
        z_hat = self.decoder(latent)
        # PREDICTION
        y = self.predictor(uc)
        return {
            'z_hat': z_hat,
            'y_hat': y,
            'ur': ur,
            'uc': uc
        }

@dataclass(unsafe_hash=True)
class DAE_1T(nn.Module):
    dim_uc: int = 10
    dim_ur: int = 10
    dim_input: int = 32

    def __post_init__(self):
        super().__init__()
        self.dim_shared=self.dim_uc
        self.encoder = nn.Sequential(
            nn.Linear(self.dim_input,
                      (self.dim_input) // 3),
            nn.SiLU(),
            nn.Linear(
                (self.dim_input) // 3,
                self.dim_uc + self.dim_ur),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.dim_uc + self.dim_ur,
                      self.dim_uc + self.dim_ur),
            nn.SiLU(),
            nn.Linear(self.dim_uc + self.dim_ur,
                      self.dim_input))

    def forward(self, z, device="cpu"):
        latent = self.encoder(z)
        # SPLIT LATENT TO GET VARIABLES
        ur = latent[:, 0:self.dim_ur]
        uc = latent[:,
                          self.dim_ur:self.dim_ur +
                          self.dim_uc]
        # DECODE LATENT
        z_hat = self.decoder(latent)
        return {
            'z_hat': z_hat,
            'ur': ur,
            'uc': uc
        }


@dataclass(unsafe_hash=True)
class Debiasor(nn.Module):
    dim_var1: int = 64
    dim_var2: int = 2096
    dim_latent1: int = 10
    dim_latent2: int = 10
    dim_shared: int = 50

    def __post_init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.dim_var2 + self.dim_var1,
                      (self.dim_var2 + self.dim_var1) //
                      3),
            nn.SiLU(),
            nn.Linear(
                (self.dim_var2 + self.dim_var1) // 3,
                self.dim_latent2 + self.dim_shared +
                self.dim_latent1),
        )
        self.decoder_var2 = nn.Sequential(
            nn.Linear(self.dim_latent2 + self.dim_shared,
                      self.dim_latent2 + self.dim_shared),
            nn.SiLU(),
            nn.Linear(self.dim_latent2 + self.dim_shared,
                      self.dim_var2))

        self.decoder_var1 = nn.Sequential(
            nn.Linear(self.dim_latent1 + self.dim_shared,
                      self.dim_latent1 + self.dim_shared),
            nn.SiLU(),
            nn.Linear(self.dim_latent1 + self.dim_shared,
                      self.dim_var1))

    def forward(self, var1, var2, device="cpu"):
        stack = torch.hstack((var1, var2)).to(device)
        latent = self.encoder(stack)
        # SPLIT LATENT TO GET VARIABLES
        latent1 = latent[:, 0:self.dim_latent1]
        shared = latent[:,
                          self.dim_latent1:self.dim_latent1 +
                          self.dim_shared]
        latent2 = latent[:, self.dim_latent1 +
                               self.dim_shared:]
        # DECODE LATENT
        var1_reconstruct = self.decoder_var1(
            latent[:, 0:self.dim_latent1 + self.dim_shared])
        var2_reconstruct = self.decoder_var2(
            latent[:, self.dim_latent1:])
        # END FORWARD
        return {
            'var1_hat': var1_reconstruct,
            'var2_hat': var2_reconstruct,
            'latent1': latent1,
            'latent2': latent2,
            'shared': shared
        }


@dataclass(unsafe_hash=True)
class Debiasor_MASK(nn.Module):
    dim_teacher_latent: int = 64
    dim_conceptual_space: int = 2096
    dim_teacher_noise = 10
    dim_conceptual_noise = 10

    def __post_init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(self.dim_conceptual_space + self.dim_teacher_latent,
                      (self.dim_conceptual_space + self.dim_teacher_latent) //
                      3),
            nn.SiLU(),
            nn.Linear(
                (self.dim_conceptual_space + self.dim_teacher_latent) // 3,
                self.dim_conceptual_noise + self.dim_conceptual_space +
                self.dim_teacher_noise),
        )
        self.decoder_conceptual = nn.Sequential(
            nn.Linear(self.dim_conceptual_noise + self.dim_conceptual_space,
                      self.dim_conceptual_noise + self.dim_conceptual_space),
            nn.SiLU(),
            nn.Linear(self.dim_conceptual_noise + self.dim_conceptual_space,
                      self.dim_conceptual_space))

        self.decoder_teacher = nn.Sequential(
            nn.Linear(self.dim_conceptual_noise + self.dim_conceptual_space,
                      self.dim_conceptual_noise + self.dim_conceptual_space),
            nn.SiLU(),
            nn.Linear(self.dim_teacher_noise + self.dim_conceptual_space,
                      self.dim_teacher_latent))

    def forward(self, z, psi, device="cpu"):
        stack = torch.hstack((psi, z)).to(device)
        latent = self.encoder(stack)
        # SPLIT LATENT TO GET VARIABLES
        concept_noise = latent[:, 0:self.dim_conceptual_noise]
        mask = torch.sigmoid(
            latent[:, self.dim_conceptual_noise:self.dim_conceptual_space +
                   self.dim_conceptual_noise])
        concepts = mask * psi
        teacher_noise = latent[:, self.dim_conceptual_noise +
                               self.dim_conceptual_space:]
        # DECODE LATENT
        in_concept_decoder = torch.hstack(
            (latent[:, 0:self.dim_conceptual_noise], concepts)).to(device)
        conceptual_reconstruct = self.decoder_conceptual(in_concept_decoder)
        in_teacher_decoder = torch.hstack(
            (concepts, latent[:, self.dim_conceptual_noise +
                              self.dim_conceptual_space:])).to(device)
        teacher_reconstruct = self.decoder_teacher(in_teacher_decoder)
        # END FORWARD
        return {
            'fr_hat': teacher_reconstruct,
            'c_hat': conceptual_reconstruct,
            'b_fr': teacher_noise,
            'b_c': concept_noise,
            'm': mask,
            'c_star': concepts
        }

class Debiasor_CNN(nn.Module):
    enc: Any = None
    dim_latent: int = 10
    dim_latent_shared: int=10
    
    def __init__(self, enc = None, dim_latent = 10, dim_latent_shared = 10):
        super().__init__()
        # ATTRIBUTES
        self.dim_latent_shared = dim_latent_shared
        self.dim_latent = dim_latent
        self.encoder = enc
        
        # ARCHITECTURE
        z_enc = self.dim_latent + self.dim_latent_shared
        z = 2*self.dim_latent + self.dim_latent_shared
        if self.encoder is None:
            self.encoder = ResNet18Enc(z_dim=z_enc)
        else:
            self.encoder.fc = nn.Sequential(
                    self.encoder.fc[0],
                    nn.SiLU(),
                    nn.Linear(
                        self.encoder.fc[0].out_features,
                        z_enc
                        )
                    )
        self.decoder = ResNet18Dec(z_dim = z_enc)
        self.stacker = nn.Sequential(
                nn.Linear(2*z_enc, 3*z_enc),
                nn.SiLU(),
                nn.Linear(3*z_enc, z)
                )

    def forward(self, x1, x2, device='cpu'):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        latent = self.stacker(torch.hstack((z1, z2)).to(device))
        # EXPLOIT LATENT
        latent2 = latent[:, 0:self.dim_latent]
        shared = latent[:,
                          self.dim_latent:self.dim_latent_shared +
                          self.dim_latent]
        latent1 = latent[:, self.dim_latent +
                               self.dim_latent_shared:]
        # DECODE LATENT
        var2_reconstruct = self.decoder(
            latent[:, 0:self.dim_latent + self.dim_latent_shared])
        var1_reconstruct = self.decoder(
            torch.hstack((latent1, shared)).to(device))
        # END FORWARD
        return {
            'var1_hat': var1_reconstruct,
            'var2_hat': var2_reconstruct,
            'latent1': latent1,
            'latent2': latent2,
            'shared': shared
        }

                
    

