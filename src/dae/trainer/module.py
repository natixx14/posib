from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from dae.models.hsic import HSIC, pairwise_distances
from dae.trainer.abstract import Trainer


def cheapLIP(h, z, p=8, device="cpu"):
    '''
    INPUTS
        h: [tensor] batched images of z
        z: [tensor] batches of inputs 
        p: [int] vector norm order
    OUTPUTS
        M: [float] Approximation of the max for cheap Lipschitz constant estimation
    '''
    num = pairwise_distances(h)
    den = pairwise_distances(z) + torch.eye(z.shape[0], device=device, requires_grad=True)
    return torch.linalg.vector_norm(num/den, ord=p)

def NCELoss(x,y, beta=1):
    dist = pairwise_distances(x)
    l = torch.tensor(0).float().to(x.device)
    for el in set(y.detach().cpu().numpy()):
        m = torch.where(y==el, 1, 0).float().reshape(-1, 1)
        m_neg = torch.abs(1-m).float().reshape(-1,1)
        l += (dist@m).sum()/((beta*m.T@(dist@m_neg)).sum()+1)
    return l

class DAE_1T_LIP_Trainer(Trainer):
    
    def __init__(
            self,
            model,
            device,
            lr,
            r_pred=1,
            r_var_reconstruct=1,
            r_hsic=1,
            r_norm_regul=1,
            r_lip=100,
            training_ablation=1,
            ):
        super().__init__(model, device, lr, training_ablation)
        # ATTRIBUTES
        self.r_var_reconstruct = r_var_reconstruct
        self.r_hsic = r_hsic
        self.r_norm_regul = r_norm_regul
        self.r_pred = r_pred
        self.r_lip = r_lip

    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_training_loss = "training loss"
        self.key_var_reconstruct = 'var reconstruction'
        self.key_hsic = 'hsic'
        self.key_norm_regul = 'norm regularisation'
        self.key_pred = 'Prediction'
        self.key_lip = 'LIP'
        self.history[self.key_training_loss] = []
        self.history[self.key_var_reconstruct] = []
        self.history[self.key_hsic] = []
        self.history[self.key_norm_regul] = []
        self.history[self.key_pred] = []
        self.history[self.key_lip] = []
        self.history_test[self.key_training_loss] = []
        self.history_test[self.key_var_reconstruct] = []
        self.history_test[self.key_hsic] = []
        self.history_test[self.key_norm_regul] = []
        self.history_test[self.key_pred] = []
        self.history_test[self.key_lip] = []

    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_training_loss].append(self.runningLoss)
        histo[self.key_hsic].append(self.running_hsic)
        histo[self.key_var_reconstruct].append(
            self.runnning_var_reconstruct)
        histo[self.key_norm_regul].append(
                self.running_norm_regul)
        histo[self.key_pred].append(
                self.running_pred)
        histo[self.key_lip].append(self.running_lip)
    
    def  init_running_losses(self):
        # RUNNING LOSSES
        self.runningLoss = 0  # global
        self.runnning_var_reconstruct = 0
        self.running_hsic = 0
        self.running_norm_regul = 0
        self.running_pred = 0
        self.running_lip = 0
   
    def global_loss(self, forward_pass, z, y, l, past_z):
        l_2 = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        ## LOSSES
        l_var_reconstruction = self.r_var_reconstruct * l_2(
            forward_pass['z_hat'], z)
        if self.r_hsic:
            l_hsic = self.r_hsic * HSIC(y[:,None].float(),
                                    forward_pass['ur'],
                                    device=self.device)
        else:
            l_hsic=torch.tensor(0)
        if self.r_norm_regul:
            l_norm_regul = self.r_norm_regul*(
                torch.abs(1/forward_pass['ur']).sum() 
                )
        else:
            l_norm_regul = torch.tensor(0)
        if self.r_pred:
            l_pred = self.r_pred*ce(forward_pass['y_hat'], y)
        else:
            l_pred = torch.tensor(0)
        if self.r_lip and past_z is not None:
            ep = len(self.history[self.key_lip])+1
            # l_lip = self.r_lip*ep*(
            #        cheapLIP(forward_pass['y_hat'], z, device=self.device)/(cheapLIP(l, z, device=self.device))
            #        )
            l_lip = torch.tensor(0.)
            ord=6
            parameters = [p for p in self.model.predictor.parameters() if p.grad is not None and p.requires_grad]
            parameters += [p for p in self.model.encoder_c.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                l_lip += p.grad.data.norm(1).item()**ord
            l_lip = self.r_lip * (l_lip**1/ord) 
        else:
            l_lip = torch.tensor(0)
        # GLOBAL LOSS
        loss = l_var_reconstruction + l_hsic + l_norm_regul + l_pred #+ l_lip + self.model.predictor[0].weight.norm(1).sum()**2/self.model.predictor[0].weight.norm(2).sum()
        return {
                'loss':loss,
                'var_reconstruction': l_var_reconstruction,
                'pred': l_pred,
                'hsic': l_hsic,
                'lip': l_lip,
                'norm_regul': l_norm_regul
                }


    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        n_batch = len(data)
        past_z = None
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            # GET DATA
            l,z, y = batch
            l = l.to(self.device)
            z = z.to(self.device)
            z.requires_grad = True # need to keep trk of grad(h) wrt z
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(z, device=self.device)
            ## LOSSES
            losses = self.global_loss(forward_pass, z, y, l, past_z)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
                past_z = z
            # LOSS HISTORY
            self.runningLoss += losses['loss'].item()
            self.runnning_var_reconstruct += losses['var_reconstruction'].item(
            )
            self.running_hsic += losses['hsic'].item()
            self.running_norm_regul += losses['norm_regul'].item()
            self.running_pred += losses['pred'].item()
            self.running_lip += losses['lip'].item()
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                    f"|\tbatch {i}/{n_batch}, total loss :{self.runningLoss/n_batch:.4f}"
                )   
        self.write_history(train=train)
    
    def perfo(self, data, head):
        self.model.eval()
        n_batch=len(data)
        acc = {
                'rec_teacher':0,
                'teacher':0,
                'pred': 0,
                }
        for _,batch in enumerate(data):
            l,z,y = batch
            z = z.to(self.device)
            l = l.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(z, device=self.device)
            rec_z=forward_pass['z_hat']
            rec_l=head(rec_z)
            y_t = torch.argmax(
                    torch.nn.functional.softmax(l.detach(),dim=1),
                    dim=1
                    )
            rec_y = torch.argmax(
                    torch.nn.functional.softmax(rec_l.detach(),dim=1),
                    dim=1
                    )
            pred_y = torch.argmax(
                    torch.nn.functional.softmax(forward_pass['y_hat'].detach(), dim=1),
                    dim=1
                    )
            acc['pred'] += (pred_y==y).sum().item()
            acc['rec_teacher'] += (rec_y==y).sum().item()
            acc['teacher'] += (y_t==y).sum().item()
        n = n_batch*data.batch_size
        return {key:val/n for key,val in acc.items()}

    def ablation_study(self, data, head):
        self.model.eval()
        n_batch=len(data)
        acc = {
                'teacher':0,
                'ab_latent':0,
                'ab_shared':0,
                }
        for _,batch in enumerate(data):
            l,z,y = batch
            z = z.to(self.device)
            l = l.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(z, device=self.device)
            rec_z=forward_pass['z_hat']
            shared=forward_pass['uc']
            latent=forward_pass['ur']
            # ABLATIONS
            ab_latent = torch.hstack((
                    torch.rand(latent.shape, device=self.device)*torch.max(latent),
                    shared
                    ))
            ab_shared = torch.hstack((
                    latent,
                    torch.rand(shared.shape, device=self.device)*torch.max(shared)
                    ))
            # FORWARD ON HEAD
            l_ab_latent=head(
                    self.model.decoder(ab_latent)
                    )
            l_ab_shared=head(
                    self.model.decoder(ab_shared)
                    )
            # COMPUTE LABEL
            y_t = torch.argmax(
                    torch.nn.functional.softmax(l.detach(),dim=1),
                    dim=1
                    )
            y_ab_latent = torch.argmax(
                    torch.nn.functional.softmax(l_ab_latent.detach(),dim=1),
                    dim=1
                    )
            y_ab_shared = torch.argmax(
                    torch.nn.functional.softmax(l_ab_shared.detach(),dim=1),
                    dim=1
                    )
            acc['ab_shared'] += (y_ab_shared==y).sum().item()
            acc['ab_latent'] += (y_ab_latent==y).sum().item()
            acc['teacher'] += (y_t==y).sum().item()
        n = n_batch*data.batch_size
        return {key:val/n for key,val in acc.items()}
class DAE_1T_Trainer(Trainer):
    
    def __init__(
            self,
            model,
            device,
            lr,
            r_nce=1,
            r_var_reconstruct=1,
            r_hsic=1,
            r_norm_regul=1,
            training_ablation=1,
            ):
        super().__init__(model, device, lr, training_ablation)
        # ATTRIBUTES
        self.r_var_reconstruct = r_var_reconstruct
        self.r_hsic = r_hsic
        self.r_norm_regul = r_norm_regul
        self.r_nce = r_nce

    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_training_loss = "training loss"
        self.key_var_reconstruct = 'var reconstruction'
        self.key_hsic = 'hsic'
        self.key_norm_regul = 'norm regularisation'
        self.key_nce = 'NCE term'
        self.history[self.key_training_loss] = []
        self.history[self.key_var_reconstruct] = []
        self.history[self.key_hsic] = []
        self.history[self.key_norm_regul] = []
        self.history[self.key_nce] = []
        self.history_test[self.key_training_loss] = []
        self.history_test[self.key_var_reconstruct] = []
        self.history_test[self.key_hsic] = []
        self.history_test[self.key_norm_regul] = []
        self.history_test[self.key_nce] = []

    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_training_loss].append(self.runningLoss)
        histo[self.key_hsic].append(self.running_hsic)
        histo[self.key_var_reconstruct].append(
            self.runnning_var_reconstruct)
        histo[self.key_norm_regul].append(
                self.running_norm_regul)
        histo[self.key_nce].append(
                self.running_nce)
    
    def  init_running_losses(self):
        # RUNNING LOSSES
        self.runningLoss = 0  # global
        self.runnning_var_reconstruct = 0
        self.running_hsic = 0
        self.running_norm_regul = 0
        self.running_nce = 0
   
    def global_loss(self, forward_pass, z, y):
        l_2 = nn.MSELoss()
        ## LOSSES
        l_var_reconstruction = self.r_var_reconstruct * l_2(
            forward_pass['z_hat'], z)
        if self.r_hsic:
            l_hsic = self.r_hsic * HSIC(y[:,None].float(),
                                    forward_pass['ur'],
                                    device=self.device)
        else:
            l_hsic=torch.tensor(0)
        if self.r_norm_regul:
            l_norm_regul = self.r_norm_regul*(
                torch.abs(1/forward_pass['ur']).sum() 
                )
        else:
            l_norm_regul = torch.tensor(0)
        if self.r_nce:
            l_NCE = self.r_nce*NCELoss(forward_pass['uc'], y)
        else: 
            l_NCE = torch.tensor(0)
        # GLOBAL LOSS
        loss = l_var_reconstruction + l_hsic + l_norm_regul + l_NCE
        return {
                'loss':loss,
                'var_reconstruction': l_var_reconstruction,
                'nce': l_NCE,
                'hsic': l_hsic,
                'norm_regul': l_norm_regul
                }


    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        n_batch = len(data)
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            # GET DATA
            l,z, y = batch
            l = l.to(self.device)
            z = z.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(z, device=self.device)
            ## LOSSES
            losses = self.global_loss(forward_pass, z, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            # LOSS HISTORY
            self.runningLoss += losses['loss'].item()
            self.runnning_var_reconstruct += losses['var_reconstruction'].item(
            )
            self.running_hsic += losses['hsic'].item()
            self.running_norm_regul += losses['norm_regul'].item()
            self.running_nce += losses['nce'].item()
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                    f"|\tbatch {i}/{n_batch}, total loss :{self.runningLoss/n_batch:.4f}"
                )   
        self.write_history(train=train)
    
    def perfo(self, data, head):
        self.model.eval()
        n_batch=len(data)
        acc = {
                'rec_teacher':0,
                'teacher':0,
                }
        for _,batch in enumerate(data):
            l,z,y = batch
            z = z.to(self.device)
            l = l.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(z, device=self.device)
            rec_z=forward_pass['z_hat']
            rec_l=head(rec_z)
            y_t = torch.argmax(
                    torch.nn.functional.softmax(l.detach(),dim=1),
                    dim=1
                    )
            rec_y = torch.argmax(
                    torch.nn.functional.softmax(rec_l.detach(),dim=1),
                    dim=1
                    )
            acc['rec_teacher'] += (rec_y==y).sum().item()
            acc['teacher'] += (y_t==y).sum().item()
        n = n_batch*data.batch_size
        return {key:val/n for key,val in acc.items()}

    def ablation_study(self, data, head):
        self.model.eval()
        n_batch=len(data)
        acc = {
                'teacher':0,
                'ab_latent':0,
                'ab_shared':0,
                }
        for _,batch in enumerate(data):
            l,z,y = batch
            z = z.to(self.device)
            l = l.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(z, device=self.device)
            rec_z=forward_pass['z_hat']
            shared=forward_pass['uc']
            latent=forward_pass['ur']
            # ABLATIONS
            ab_latent = torch.hstack((
                    torch.rand(latent.shape, device=self.device)*torch.max(latent),
                    shared
                    ))
            ab_shared = torch.hstack((
                    latent,
                    torch.rand(shared.shape, device=self.device)*torch.max(shared)
                    ))
            # FORWARD ON HEAD
            l_ab_latent=head(
                    self.model.decoder(ab_latent)
                    )
            l_ab_shared=head(
                    self.model.decoder(ab_shared)
                    )
            # COMPUTE LABEL
            y_t = torch.argmax(
                    torch.nn.functional.softmax(l.detach(),dim=1),
                    dim=1
                    )
            y_ab_latent = torch.argmax(
                    torch.nn.functional.softmax(l_ab_latent.detach(),dim=1),
                    dim=1
                    )
            y_ab_shared = torch.argmax(
                    torch.nn.functional.softmax(l_ab_shared.detach(),dim=1),
                    dim=1
                    )
            acc['ab_shared'] += (y_ab_shared==y).sum().item()
            acc['ab_latent'] += (y_ab_latent==y).sum().item()
            acc['teacher'] += (y_t==y).sum().item()
        n = n_batch*data.batch_size
        return {key:val/n for key,val in acc.items()}

class DAE_Trainer(Trainer):
    
    def __init__(
            self,
            model,
            device,
            lr,
            r_nce=1,
            r_var1_reconstruct=1,
            r_var2_reconstruct=1,
            r_hsic=1,
            r_norm_regul=1,
            training_ablation=1,
            ):
        super().__init__(model, device, lr, training_ablation)
        # ATTRIBUTES
        self.r_var1_reconstruct = r_var1_reconstruct
        self.r_var2_reconstruct = r_var2_reconstruct
        self.r_hsic = r_hsic
        self.r_norm_regul = r_norm_regul
        self.r_nce = r_nce

    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_training_loss = "training loss"
        self.key_var2_reconstruct = 'var2 reconstruction'
        self.key_var1_reconstruct = 'var1 reconstruction'
        self.key_hsic = 'hsic'
        self.key_norm_regul = 'norm regularisation'
        self.key_nce = 'NCE term'
        self.history[self.key_training_loss] = []
        self.history[self.key_var2_reconstruct] = []
        self.history[self.key_var1_reconstruct] = []
        self.history[self.key_hsic] = []
        self.history[self.key_norm_regul] = []
        self.history[self.key_nce] = []
        self.history_test[self.key_training_loss] = []
        self.history_test[self.key_var2_reconstruct] = []
        self.history_test[self.key_var1_reconstruct] = []
        self.history_test[self.key_hsic] = []
        self.history_test[self.key_norm_regul] = []
        self.history_test[self.key_nce] = []

    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_training_loss].append(self.runningLoss)
        histo[self.key_hsic].append(self.running_hsic)
        histo[self.key_var1_reconstruct].append(
            self.runnning_var1_reconstruct)
        histo[self.key_var2_reconstruct].append(
            self.running_var2_reconstruct)
        histo[self.key_norm_regul].append(
                self.running_norm_regul)
        histo[self.key_nce].append(
                self.running_nce)
    
    def  init_running_losses(self):
        # RUNNING LOSSES
        self.runningLoss = 0  # global
        self.running_var2_reconstruct = 0
        self.runnning_var1_reconstruct = 0
        self.running_hsic = 0
        self.running_norm_regul = 0
        self.running_nce = 0
   
    def global_loss(self, forward_pass, u, v, y):
        l_2 = nn.MSELoss()
        ## LOSSES
        l_var2_reconstruction = self.r_var2_reconstruct * l_2(
            forward_pass['var2_hat'], v)
        l_var1_reconstruction = self.r_var1_reconstruct * l_2(
            forward_pass['var1_hat'], u)
        if self.r_hsic:
            l_hsic = self.r_hsic * HSIC(forward_pass['latent1'],
                                    forward_pass['latent2'],
                                    device=self.device)
        else:
            l_hsic=torch.tensor(0)
        if self.r_norm_regul:
            l_norm_regul = self.r_norm_regul*(
                torch.abs(1/forward_pass['latent1']).sum() + torch.abs(1 / forward_pass['latent2']).sum()
                )
        else:
            l_norm_regul = torch.tensor(0)
        if self.r_nce:
            l_NCE = self.r_nce*NCELoss(forward_pass['shared'], y)
        else: 
            l_NCE = torch.tensor(0)
        # GLOBAL LOSS
        loss = l_var2_reconstruction + l_var1_reconstruction + l_hsic + l_norm_regul + l_NCE
        return {
                'loss':loss,
                'var2_reconstruction': l_var2_reconstruction,
                'var1_reconstruction': l_var1_reconstruction,
                'nce': l_NCE,
                'hsic': l_hsic,
                'norm_regul': l_norm_regul
                }


    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        n_batch = len(data)
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            # GET DATA
            l1,u,l2,v, y = batch
            u = u.to(self.device)
            v = v.to(self.device)
            y = y.to(self.device)
            l1 = l1.to(self.device)
            l2 = l2.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(u, v, device=self.device)
            ## LOSSES
            losses = self.global_loss(forward_pass, u, v, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            # LOSS HISTORY
            self.runningLoss += losses['loss'].item()
            self.running_var2_reconstruct += losses['var2_reconstruction'].item(
            )
            self.runnning_var1_reconstruct += losses['var1_reconstruction'].item(
            )
            self.running_hsic += losses['hsic'].item()
            self.running_norm_regul += losses['norm_regul'].item()
            self.running_nce += losses['nce'].item()
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                    f"|\tbatch {i}/{n_batch}, total loss :{self.runningLoss/n_batch:.4f}"
                )   
        self.write_history(train=train)
    
    def perfo(self, data, head1, head2):
        self.model.eval()
        n_batch=len(data)
        acc = {
                'rec_teacher1':0,
                'teacher1':0,
                'rec_teacher2':0,
                'teacher2':0
                }
        for _,batch in enumerate(data):
            l1,z1,l2,z2,y = batch
            z1 = z1.to(self.device)
            z2 = z2.to(self.device)
            l1 = l1.to(self.device)
            l2 = l2.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(z1, z2, device=self.device)
            rec_z1=forward_pass['var1_hat']
            rec_z2=forward_pass['var2_hat']
            rec_l1=head1(rec_z1)
            rec_l2=head2(rec_z2)
            y1 = torch.argmax(
                    torch.nn.functional.softmax(l1.detach(),dim=1),
                    dim=1
                    )
            y2 = torch.argmax(
                    torch.nn.functional.softmax(l2.detach(),dim=1),
                    dim=1
                    )
            rec_y1 = torch.argmax(
                    torch.nn.functional.softmax(rec_l1.detach(),dim=1),
                    dim=1
                    )
            rec_y2 = torch.argmax(
                    torch.nn.functional.softmax(rec_l2.detach(),dim=1),
                    dim=1
                    )
            acc['rec_teacher1'] += (rec_y1==y).sum().item()
            acc['rec_teacher2'] += (rec_y2==y).sum().item()
            acc['teacher1'] += (y1==y).sum().item()
            acc['teacher2'] += (y2==y).sum().item()
        n = n_batch*data.batch_size
        return {key:val/n for key,val in acc.items()}

    def ablation_study(self, data, head1, head2):
        self.model.eval()
        n_batch=len(data)
        acc = {
                'teacher1':0,
                'ab_latent1':0,
                'ab_shared1':0,
                'teacher2':0,
                'ab_latent2':0,
                'ab_shared2':0
                }
        for _,batch in enumerate(data):
            l1,z1,l2,z2,y = batch
            z1 = z1.to(self.device)
            z2 = z2.to(self.device)
            l1 = l1.to(self.device)
            l2 = l2.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(z1, z2, device=self.device)
            rec_z1=forward_pass['var1_hat']
            rec_z2=forward_pass['var2_hat']
            shared=forward_pass['shared']
            latent1=forward_pass['latent1']
            latent2=forward_pass['latent2']
            # ABLATIONS
            ab_latent1 = torch.hstack((
                    torch.rand(latent1.shape, device=self.device)*torch.max(latent1),
                    shared
                    ))
            ab_shared1 = torch.hstack((
                    latent1,
                    torch.rand(shared.shape, device=self.device)*torch.max(shared)
                    ))
            ab_latent2 = torch.hstack((
                    shared,
                    torch.rand(latent2.shape, device=self.device)*torch.max(latent2)
                    ))
            ab_shared2 = torch.hstack((
                    torch.rand(shared.shape, device=self.device)*torch.max(shared),
                    latent2
                    ))
            # FORWARD ON HEAD
            l_ab_latent1=head1(
                    self.model.decoder_var1(ab_latent1)
                    )
            l_ab_shared1=head1(
                    self.model.decoder_var1(ab_shared1)
                    )
            l_ab_latent2=head2(
                    self.model.decoder_var2(ab_latent2)
                    )
            l_ab_shared2=head2(
                    self.model.decoder_var2(ab_shared2)
                    )
            # COMPUTE LABEL
            y1 = torch.argmax(
                    torch.nn.functional.softmax(l1.detach(),dim=1),
                    dim=1
                    )
            y2 = torch.argmax(
                    torch.nn.functional.softmax(l2.detach(),dim=1),
                    dim=1
                    )
            y1_ab_latent = torch.argmax(
                    torch.nn.functional.softmax(l_ab_latent1.detach(),dim=1),
                    dim=1
                    )
            y1_ab_shared = torch.argmax(
                    torch.nn.functional.softmax(l_ab_shared1.detach(),dim=1),
                    dim=1
                    )
            y2_ab_latent = torch.argmax(
                    torch.nn.functional.softmax(l_ab_latent2.detach(),dim=1),
                    dim=1
                    )
            y2_ab_shared = torch.argmax(
                    torch.nn.functional.softmax(l_ab_shared2.detach(),dim=1),
                    dim=1
                    )
            acc['ab_shared1'] += (y1_ab_shared==y).sum().item()
            acc['ab_latent1'] += (y1_ab_latent==y).sum().item()
            acc['ab_shared2'] += (y2_ab_shared==y).sum().item()
            acc['ab_latent2'] += (y2_ab_latent==y).sum().item()
            acc['teacher1'] += (y1==y).sum().item()
            acc['teacher2'] += (y2==y).sum().item()
        n = n_batch*data.batch_size
        return {key:val/n for key,val in acc.items()}


class DAE_NCE_Trainer(DAE_Trainer):

    def __init__(
            self,
            model,
            device,
            lr,
            r_nce=1,
            r_var1_reconstruct=1,
            r_var2_reconstruct=1,
            r_hsic=1,
            r_norm_regul=1,
            training_ablation=1,
            ):
        super().__init__(model, device, lr, training_ablation)
        # ATTRIBUTES
        self.r_var1_reconstruct = r_var1_reconstruct
        self.r_var2_reconstruct = r_var2_reconstruct
        self.r_hsic = r_hsic
        self.r_norm_regul = r_norm_regul
        self.r_nce = r_nce

    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_training_loss = "training loss"
        self.key_var2_reconstruct = 'var2 reconstruction'
        self.key_var1_reconstruct = 'var1 reconstruction'
        self.key_hsic = 'hsic'
        self.key_norm_regul = 'norm regularisation'
        self.key_nce = 'NCE term'
        self.history[self.key_training_loss] = []
        self.history[self.key_var2_reconstruct] = []
        self.history[self.key_var1_reconstruct] = []
        self.history[self.key_hsic] = []
        self.history[self.key_norm_regul] = []
        self.history[self.key_nce] = []
        self.history_test[self.key_training_loss] = []
        self.history_test[self.key_var2_reconstruct] = []
        self.history_test[self.key_var1_reconstruct] = []
        self.history_test[self.key_hsic] = []
        self.history_test[self.key_norm_regul] = []
        self.history_test[self.key_nce] = []

    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_training_loss].append(self.runningLoss)
        histo[self.key_hsic].append(self.running_hsic)
        histo[self.key_var1_reconstruct].append(
            self.runnning_var1_reconstruct)
        histo[self.key_var2_reconstruct].append(
            self.running_var2_reconstruct)
        histo[self.key_norm_regul].append(
                self.running_norm_regul)
        histo[self.key_nce].append(
                self.running_nce)
    
    def  init_running_losses(self):
        # RUNNING LOSSES
        self.runningLoss = 0  # global
        self.running_var2_reconstruct = 0
        self.runnning_var1_reconstruct = 0
        self.running_hsic = 0
        self.running_norm_regul = 0
        self.running_nce = 0
   
    def global_loss(self, forward_pass, u, v, y):
        l_2 = nn.MSELoss()
        ## LOSSES
        l_var2_reconstruction = self.r_var2_reconstruct * l_2(
            forward_pass['var2_hat'], v)
        l_var1_reconstruction = self.r_var1_reconstruct * l_2(
            forward_pass['var1_hat'], u)
        l_hsic = self.r_hsic * HSIC(forward_pass['latent1'],
                                    forward_pass['latent2'],
                                    device=self.device)
        l_norm_regul = self.r_norm_regul*(
                torch.abs(1/forward_pass['latent1']).sum() + torch.abs(1 / forward_pass['latent2']).sum()
                ) 

        l_NCE = self.r_nce*NCELoss(forward_pass['shared'], y)
        # GLOBAL LOSS
        loss = l_var2_reconstruction + l_var1_reconstruction + l_hsic + l_norm_regul
        return {
                'loss':loss,
                'var2_reconstruction': l_var2_reconstruction,
                'var1_reconstruction': l_var1_reconstruction,
                'hsic': l_hsic,
                'nce': l_NCE,
                'norm_regul': l_norm_regul
                }


    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        l_2 = nn.MSELoss()
        n_batch = len(data)
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            # GET DATA
            u, v, y = batch
            u = u.to(self.device)
            v = v.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(u, v, device=self.device)
            ## LOSSES
            losses = self.global_loss(forward_pass, u, v, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            # LOSS HISTORY
            self.runningLoss += losses['loss'].item()
            self.running_var2_reconstruct += losses['var2_reconstruction'].item(
            )
            self.runnning_var1_reconstruct += losses['var1_reconstruction'].item(
            )
            self.running_hsic += losses['hsic'].item()
            self.running_norm_regul += losses['norm_regul'].item()
            self.running_nce += losses['nce'].item()
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                    f"|\tbatch {i}/{n_batch}, total loss :{self.runningLoss/n_batch:.4f}"
                )   
        self.write_history(train=train)





class DAE_CB2_Trainer(Trainer):

    def __init__(
            self,
            f,
            model,
            device,
            lr,
            r_conceptual_reconstruct=1,
            r_teacher_reconstruct=1,
            r_hsic=1,
            training_ablation=1,
            r_rappel=1,
            r_nce=1,
            r_norm_regul=1
            ):
        super().__init__(model, device, lr, training_ablation)
        # ATTRIBUTES
        self.f = f
        self.r_conceptual_reconstruct = r_conceptual_reconstruct
        self.r_teacher_reconstruct = r_teacher_reconstruct
        self.r_hsic = r_hsic
        self.r_nce = r_nce
        self.r_norm_regul = r_norm_regul
        ## PREPARE TEACHER MODEL
        self.latent_f = torch.nn.Sequential(*(list(self.f.children())[:-1]))
        self.f.eval()
        self.latent_f.eval()
        self.f.requires_grad_(False)
        self.latent_f.requires_grad_(False)

    def history_init(self):
        self.history = {}
        self.key_training_loss = "training loss"
        self.key_conceptual_reconstruct = 'conceptual reconstruction'
        self.key_teacher_reconstruct = 'teacher reconstruction'
        self.key_hsic = 'hsic'
        self.key_norm_regul = 'norm regularisation'
        self.key_nce = 'NCE loss'
        self.history[self.key_training_loss] = []
        self.history[self.key_conceptual_reconstruct] = []
        self.history[self.key_teacher_reconstruct] = []
        self.history[self.key_hsic] = []
        self.history[self.key_norm_regul] = []
        self.history[self.key_nce] = []

    def write_history(self):
        self.history[self.key_training_loss].append(self.runningLoss)
        self.history[self.key_hsic].append(self.running_hsic)
        self.history[self.key_teacher_reconstruct].append(
            self.runnning_teacher_reconstruct)
        self.history[self.key_conceptual_reconstruct].append(
            self.running_concept_reconstruct)
        self.history[self.key_norm_regul].append(
                self.running_norm_regul)
        self.history[self.key_nce].append(
                self.running_nce)
    
    def  init_running_losses(self):
        # RUNNING LOSSES
        self.runningLoss = 0  # global
        self.running_concept_reconstruct = 0
        self.runnning_teacher_reconstruct = 0
        self.running_hsic = 0
        self.running_norm_regul = 0
        self.running_nce = 0

    def global_loss(self, forward_pass, x, psi, latent_f, y):
        l_2 = nn.MSELoss()
        ## LOSSES
        l_conceptual_reconstruction = self.r_conceptual_reconstruct * l_2(
            forward_pass['c_hat'], psi)
        l_teacher_reconstruction = self.r_teacher_reconstruct * l_2(
            forward_pass['fr_hat'], latent_f)
        l_hsic = self.r_hsic * HSIC(forward_pass['b_c'],
                                    forward_pass['b_fr'],
                                    device=self.device)
        l_NCE = self.r_nce*NCELoss(forward_pass['c_star'], y)
        l_norm_regul = self.r_norm_regul*torch.abs(1/forward_pass['b_fr']).sum() + torch.abs(1 / forward_pass['b_c']).sum() 
        # GLOBAL LOSS
        loss = l_conceptual_reconstruction + l_teacher_reconstruction + l_hsic + l_norm_regul + l_NCE
        #+forward_pass['m'].sum() #+ (torch.square(torch.sum(torch.abs(forward_pass['m']),dim=1)) / torch.sum(torch.square(forward_pass['m']), dim=1)).sum()
        return {
                'loss':loss,
                'conceptual_reconstruction': l_conceptual_reconstruction,
                'teacher_reconstruction': l_teacher_reconstruction,
                'hsic': l_hsic,
                'norm_regul': l_norm_regul,
                'nce': l_NCE
                }


    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        l_2 = nn.MSELoss()
        n_batch = len(data)
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            # GET DATA
            x, psi, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            # FORWARD ON TEACHER
            latent_f = self.latent_f(x).view(x.size(0), -1)
            f_out = self.f(x)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(latent_f, psi, device=self.device)
            ## LOSSES
            losses = self.global_loss(forward_pass, x, psi, latent_f, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
                # LOSS HISTORY
                self.runningLoss += losses['loss'].item()
                self.running_concept_reconstruct += losses['conceptual_reconstruction'].item(
                )
                self.runnning_teacher_reconstruct += losses['teacher_reconstruction'].item(
                )
                self.running_hsic += losses['hsic'].item()
                self.running_norm_regul += losses['norm_regul'].item()
                self.running_nce += losses['nce'].item()
            else:
                pass
            if i % (n_batch // 5) == n_batch // 5 - 1:
                self.logger.info(
                    f"|\tbatch {i}/{n_batch}, total loss :{self.runningLoss/n_batch:.4f}"
                )

class Trainer_stacked(Trainer):

    def __init__(
            self,
            model,
            r_pred,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lr=1e-3,
            training_ablation=1
            ):
        super().__init__(model, device, lr, training_ablation)
        self.r_pred = r_pred
    
    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_loss_global = "global loss"
        self.key_accuracy = "accuracy"
        keys = [
                self.key_loss_global,
                self.key_accuracy
                ]
        for key in keys:
            self.history[key] = []
            self.history_test[key] = []

    def  init_running_losses(self):
        # RUNNING LOSSES
        self.running_loss_global = 0  # global
        # RUNNING ACC
        self.running_accuracy = 0
 
    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_loss_global].append(self.running_loss_global)    
        histo[self.key_accuracy].append(self.running_accuracy)

    def fit_verbose(self, key_list, ep, epochs):
        if not key_list:
            key_list = self.history.keys()
        if not ep:
            self.logger.info(f"=== Epoch {ep+1}/{epochs}")
        else:
            self.logger.info(
                    f'=== Epoch {ep+1}/{epochs} '
                    + 
                    ' '.join([f'{k} : {self.history[k][-1]:.3f}' for k in key_list])
                    +
                    f" test accuracy: {self.history_test['accuracy'][-1]-self.history['accuracy'][-1]:.3f}"
                    )

    def global_loss(self, forward_pass, y):
        l_ce = nn.CrossEntropyLoss()
        # PREDICTION
        l_pred = self.r_pred * l_ce(forward_pass, y)
        # GLOBAL LOSS
        loss = l_pred
        # ACCURACY
        y_f = torch.argmax(torch.nn.functional.softmax(forward_pass.detach(),dim=1),dim=1)
        accuracy = (y_f == y).sum().item()
        return {
                'loss':loss,
                'accuracy': accuracy,
                }
    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        n_batch = len(data)
        n = n_batch * data.batch_size
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            # GET DATA
            l1, _,l2, _, y = batch
            l1 = l1.to(self.device)
            l2 = l2.to(self.device)
            y = y.to(self.device)
            # FORWARD PASSES
            forward_pass = self.model(torch.hstack((l1,l2)).to(self.device))
            ## LOSSES
            losses = self.global_loss(forward_pass, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            # LOSS HISTORY
            self.running_loss_global += losses['loss'].item()/n
            # ACCURACY HISTORY
            self.running_accuracy += losses['accuracy']/n
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                        f"|\tbatch {i}/{n_batch}, total loss :{self.running_loss_global/n_batch:.4f} acc :{self.running_accuracy:.4f}"
                )   
        self.write_history(train=train)

class Trainer_shared(Trainer):

    def __init__(
            self,
            model,
            dae,
            r_pred,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lr=1e-3,
            training_ablation=1
            ):
        super().__init__(model, device, lr, training_ablation)
        self.dae = dae
        self.dae.requires_grad_(False)
        self.r_pred = r_pred
    
    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_loss_global = "global loss"
        self.key_accuracy = "accuracy"
        keys = [
                self.key_loss_global,
                self.key_accuracy
                ]
        for key in keys:
            self.history[key] = []
            self.history_test[key] = []

    def  init_running_losses(self):
        # RUNNING LOSSES
        self.running_loss_global = 0  # global
        # RUNNING ACC
        self.running_accuracy = 0
 
    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_loss_global].append(self.running_loss_global)    
        histo[self.key_accuracy].append(self.running_accuracy)

    def fit_verbose(self, key_list, ep, epochs):
        if not key_list:
            key_list = self.history.keys()
        if not ep:
            self.logger.info(f"=== Epoch {ep+1}/{epochs}")
        else:
            self.logger.info(
                    f'=== Epoch {ep+1}/{epochs} '
                    + 
                    ' '.join([f'{k} : {self.history[k][-1]:.3f}' for k in key_list])
                    +
                    f" test accuracy: {self.history_test['accuracy'][-1]-self.history['accuracy'][-1]:.3f}"
                    )

    def global_loss(self, forward_pass, y):
        l_ce = nn.CrossEntropyLoss()
        # PREDICTION
        l_pred = self.r_pred * l_ce(forward_pass, y)
        # GLOBAL LOSS
        loss = l_pred
        # ACCURACY
        y_f = torch.argmax(torch.nn.functional.softmax(forward_pass.detach(),dim=1),dim=1)
        accuracy = (y_f == y).sum().item()
        return {
                'loss':loss,
                'accuracy': accuracy,
                }
    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        n_batch = len(data)
        n = n_batch * data.batch_size
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            # GET DATA
            if len(batch)>3:
                l1, z1,l2, z2, y = batch
                z1 = z1.to(self.device)
                z2 = z2.to(self.device)
                y = y.to(self.device)
                # FORWARD PASSES
                dae_pass = self.dae(z1, z2, device=self.device)
                forward_pass = self.model(dae_pass['shared'])
            else:
                l, z, y = batch
                z = z.to(self.device)
                y = y.to(self.device)
                dae_pass = self.dae(z, device=self.device)
                forward_pass = self.model(dae_pass['uc'])
            ## LOSSES
            losses = self.global_loss(forward_pass, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            # LOSS HISTORY
            self.running_loss_global += losses['loss'].item()/n
            # ACCURACY HISTORY
            self.running_accuracy += losses['accuracy']/n
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                        f"|\tbatch {i}/{n_batch}, total loss :{self.running_loss_global/n_batch:.4f} acc :{self.running_accuracy:.4f}"
                )   
        self.write_history(train=train)

class Trainer_surrogate(Trainer):

    def __init__(
            self,
            model,
            teacher,
            dae,
            r_pred,
            r_uc,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lr=1e-3,
            training_ablation=1
            ):
        super().__init__(model, device, lr, training_ablation)
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.dae = dae
        self.dae.requires_grad_(False)
        self.r_pred = r_pred
        self.r_uc = r_uc
    
    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_loss_global = "global loss"
        self.key_distillation = "distillation"
        self.key_alignement = "alignement"
        keys = [
                self.key_loss_global,
                self.key_distillation,
                self.key_alignement
                ]
        for key in keys:
            self.history[key] = []
            self.history_test[key] = []

    def  init_running_losses(self):
        # RUNNING LOSSES
        self.running_distillation = 0  # global
        self.running_alignement = 0  # global
        self.running_loss_global = 0  # global
 
    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_loss_global].append(self.running_loss_global)    
        histo[self.key_distillation].append(self.running_distillation)    
        histo[self.key_alignement].append(self.running_alignement)    

    def fit_verbose(self, key_list, ep, epochs):
        if not key_list:
            key_list = self.history.keys()
        if not ep:
            self.logger.info(f"=== Epoch {ep+1}/{epochs}")
        else:
            self.logger.info(
                    f'=== Epoch {ep+1}/{epochs} '
                    + 
                    ' '.join([f'{k} : {self.history[k][-1]:.3f}' for k in key_list])
                    )

    def global_loss(self, forward_pass, uc, y):
        l_2 = nn.MSELoss()
        l_ce = nn.CrossEntropyLoss()
        # PREDICTION
        l_pred = self.r_pred * l_ce(forward_pass['logits'], y)
        # REPRESENTATION
        l_uc = self.r_uc * l_2(forward_pass['uc'], uc)

        # GLOBAL LOSS
        l_alignement = l_uc
        l_distill = l_pred
        loss = l_pred + l_uc
        return {
                'loss':loss,
                'distillation':l_distill,
                'alignement': l_alignement
                }

    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        n_batch = len(data)
        n = n_batch * data.batch_size
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            x, uc, y = batch
            x = x.to(self.device)
            x.requires_grad = True
            uc = uc.to(self.device)
            y = y.to(self.device)
            # FORWARD PASSES
            forward_pass = self.model(x)
            ## LOSSES
            losses = self.global_loss(forward_pass, uc, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            # LOSS HISTORY
            self.running_distillation += losses['distillation'].item()/n
            self.running_alignement += losses['alignement'].item()/n
            self.running_loss_global += losses['loss'].item()/n
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                        f"|\tbatch {i}/{n_batch}, total loss :{self.running_loss_global/n_batch:.4f}"
                )   
        self.write_history(train=train)

    def perfo(self, data):
        self.model.eval()
        n_batch=len(data)
        acc = {
                'pred': 0,
                }
        for _,batch in enumerate(data):
            x, uc, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            fp = self.model(x)
            pred_y = torch.argmax(
                    torch.nn.functional.softmax(fp['logits'].detach(), dim=1),
                    dim=1
                    )
            acc['pred'] += (pred_y==y).sum().item()
        n = n_batch*data.batch_size
        return {key:val/n for key,val in acc.items()}


class Trainer_residual(Trainer):

    def __init__(
            self,
            model,
            dae,
            r_pred,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lr=1e-3,
            training_ablation=1
            ):
        super().__init__(model, device, lr, training_ablation)
        self.dae = dae
        self.dae.requires_grad_(False)
        self.r_pred = r_pred
    
    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_loss_global = "global loss"
        self.key_accuracy = "accuracy"
        keys = [
                self.key_loss_global,
                self.key_accuracy
                ]
        for key in keys:
            self.history[key] = []
            self.history_test[key] = []

    def  init_running_losses(self):
        # RUNNING LOSSES
        self.running_loss_global = 0  # global
        # RUNNING ACC
        self.running_accuracy = 0
 
    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_loss_global].append(self.running_loss_global)    
        histo[self.key_accuracy].append(self.running_accuracy)

    def fit_verbose(self, key_list, ep, epochs):
        if not key_list:
            key_list = self.history.keys()
        if not ep:
            self.logger.info(f"=== Epoch {ep+1}/{epochs}")
        else:
            self.logger.info(
                    f'=== Epoch {ep+1}/{epochs} '
                    + 
                    ' '.join([f'{k} : {self.history[k][-1]:.3f}' for k in key_list])
                    +
                    f" test accuracy: {self.history_test['accuracy'][-1]-self.history['accuracy'][-1]:.3f}"
                    )

    def global_loss(self, forward_pass, y):
        l_ce = nn.CrossEntropyLoss()
        # PREDICTION
        l_pred = self.r_pred * l_ce(forward_pass, y)
        # GLOBAL LOSS
        loss = l_pred
        # ACCURACY
        y_f = torch.argmax(torch.nn.functional.softmax(forward_pass.detach(),dim=1),dim=1)
        accuracy = (y_f == y).sum().item()
        return {
                'loss':loss,
                'accuracy': accuracy,
                }
    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        n_batch = len(data)
        n = n_batch * data.batch_size
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            if len(batch)>3:
                l1, z1,l2, z2, y = batch
                z1 = z1.to(self.device)
                z2 = z2.to(self.device)
                y = y.to(self.device)
                # FORWARD PASSES
                dae_pass = self.dae(z1, z2, device=self.device)
                forward_pass = self.model(dae_pass['latent1'])
            else:
                l, z, y = batch
                z = z.to(self.device)
                y = y.to(self.device)
                dae_pass = self.dae(z, device=self.device)
                forward_pass = self.model(dae_pass['ur'])
            ## LOSSES
            losses = self.global_loss(forward_pass, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            # LOSS HISTORY
            self.running_loss_global += losses['loss'].item()/n
            # ACCURACY HISTORY
            self.running_accuracy += losses['accuracy']/n
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                        f"|\tbatch {i}/{n_batch}, total loss :{self.running_loss_global/n_batch:.4f} acc :{self.running_accuracy:.4f}"
                )   
        self.write_history(train=train)

class Trainer_BB(Trainer):

    def __init__(
            self,
            model,
            r_pred,
            transform,
            in_feature_fc=512,
            n_classes=10,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lr=1e-3,
            training_ablation=1
            ):
        model.fc = torch.nn.Sequential(
                    torch.nn.Linear(in_features=in_feature_fc, out_features=in_feature_fc, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_features=in_feature_fc, out_features=n_classes, bias=True),
                    )
        super().__init__(model, device, lr, training_ablation)
        self.r_pred = r_pred
        self.transform=transform
    
    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_loss_global = "global loss"
        self.key_accuracy = "accuracy"
        keys = [
                self.key_loss_global,
                self.key_accuracy
                ]
        for key in keys:
            self.history[key] = []
            self.history_test[key] = []

    def  init_running_losses(self):
        # RUNNING LOSSES
        self.running_loss_global = 0  # global
        # RUNNING ACC
        self.running_accuracy = 0
 
    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_loss_global].append(self.running_loss_global)    
        histo[self.key_accuracy].append(self.running_accuracy)

    def global_loss(self, forward_pass, y):
        l_ce = nn.CrossEntropyLoss()
        # PREDICTION
        l_pred = self.r_pred * l_ce(forward_pass, y)
        # GLOBAL LOSS
        loss = l_pred
        # ACCURACY
        y_f = torch.argmax(torch.nn.functional.softmax(forward_pass.detach(),dim=1),dim=1)
        accuracy = (y_f == y).sum().item()
        return {
                'loss':loss,
                'accuracy': accuracy,
                }
    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        n_batch = len(data)
        n = n_batch * data.batch_size
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            # GET DATA
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(x)
            ## LOSSES
            losses = self.global_loss(forward_pass, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            # LOSS HISTORY
            self.running_loss_global += losses['loss'].item()/n
            # ACCURACY HISTORY
            self.running_accuracy += losses['accuracy']/n
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                        f"|\tbatch {i}/{n_batch}, total loss :{self.running_loss_global/n_batch:.4f} acc :{self.running_accuracy/n:.4f}"
                )   
        self.write_history(train=train)
 
