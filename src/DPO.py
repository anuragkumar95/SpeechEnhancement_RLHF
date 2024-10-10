#-*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.CMGAN.actor import TSCNet
import os
from data.dataset import load_data
from data_sampler import DataSampler
import torch.nn.functional as F
import torch
from utils import power_compress, batch_pesq, copy_weights, freeze_layers

import argparse
import wandb
import numpy as np
#import traceback
from speech_enh_env import  SpeechEnhancementAgent
import torch
import wandb
#import copy
from torch.distributions import Normal

from compute_metrics import compute_metrics

import torch.multiprocessing as mp
import torch.nn.functional as F

torch.manual_seed(123)


"""
TODO:
1. Add wandb logs.
2. Add validation loop
3. Implement proper argparsing.
"""

class DPO:
    def __init__(self,
                 sft_model,
                 model,   
                 gpu_id=None, 
                 beta=0.2,):
        
        self.ref_model = sft_model
        self.ref_model.eval()
        self.model = model
        self.gpu_id = gpu_id 
        self.std = 0.01
        self.beta = beta


    def get_logprob(self, mu, x):
        std = (torch.ones(mu.shape) * self.std).to(self.gpu_id)
        N = Normal(mu, std)
        x_logprob = N.log_prob(x)
        return x_logprob 

    def dpo_loss(self, x, ypos, yneg):
        ref_mu = self.ref_model(x)
        y_mu = self.model(x)

        ref_mu = torch.cat([ref_mu[0], ref_mu[1]], dim=1)
        y_mu = torch.cat([y_mu[0], y_mu[1]], dim=1)

        print(f"REF:{ref_mu.shape}, Y:{y_mu.shape}")

        ref_pos_logprob = self.get_logprob(ref_mu, ypos)
        ref_neg_logprob = self.get_logprob(ref_mu, yneg)

        y_pos_logprob = self.get_logprob(y_mu, ypos)
        y_neg_logprob = self.get_logprob(y_mu, yneg)

        scores = self.beta * ((y_pos_logprob - ref_pos_logprob) - (y_neg_logprob - ref_neg_logprob)) 
        log_scores = torch.log(F.sigmoid(scores)).mean()

        return -log_scores

    
    def spec(self, noisy, ypos, yneg, n_fft=400, hop=100):
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy = torch.transpose(noisy, 0, 1)
        ypos = torch.transpose(ypos, 0, 1) 
        yneg = torch.transpose(yneg, 0, 1)
        noisy = torch.transpose(noisy * c, 0, 1)
        ypos = torch.transpose(ypos * c, 0, 1)
        yneg = torch.transpose(yneg * c, 0, 1)

        noisy_spec = torch.stft(
            noisy,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).to(self.gpu_id),
            onesided=True,
        )
        ypos_spec = torch.stft(
            ypos,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).to(self.gpu_id),
            onesided=True,
        )
        yneg_spec = torch.stft(
            yneg,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).to(self.gpu_id),
            onesided=True,
        )

        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        ypos_spec = power_compress(ypos_spec)
        yneg_spec = power_compress(yneg_spec)
        
        return noisy_spec, ypos_spec, yneg_spec


    def forward_step(self, x, ypos, yneg):
        if self.gpu_id is not None:
            x = x.to(self.gpu_id)
            ypos = ypos.to(self.gpu_id)
            yneg = yneg.to(self.gpu_id)

        x, ypos, yneg = self.spec(x, ypos, yneg)
        dpo_loss = self.dpo_loss(x, ypos, yneg)
        return dpo_loss


class DPOTrainer:
    def __init__(self,
                 train_ds,
                 test_ds, 
                 args, 
                 gpu_id):
        self.args = args
        self.actor = TSCNet(num_channel=64, 
                            num_features=self.args.n_fft // 2 + 1, 
                            gpu_id=gpu_id,
                            eval=True)
        self.expert = TSCNet(num_channel=64, 
                            num_features=self.args.n_fft // 2 + 1,
                            gpu_id=gpu_id,
                            eval=True)
        
        expert_checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))

        try:
            self.actor.load_state_dict(expert_checkpoint['generator_state_dict']) 
            self.expert.load_state_dict(expert_checkpoint['generator_state_dict'])

        except KeyError as e:
            self.actor.load_state_dict(expert_checkpoint)
            self.expert.load_state_dict(expert_checkpoint)
        
        #Set expert to eval and freeze all layers.
        self.expert = freeze_layers(self.expert, 'all')
        
        del expert_checkpoint 
        print(f"Loaded checkpoint stored at {args.ckpt}.")
        
        if gpu_id is not None:
            self.actor = self.actor.to(gpu_id)
            self.expert = self.expert.to(gpu_id)

        self.optimizer = torch.optim.AdamW(
            filter(lambda layer:layer.requires_grad, self.actor.parameters()), lr=args.init_lr
        )     

        self.DPO = DPO(sft_model=self.expert,
                       model=self.actor,   
                       gpu_id=gpu_id, 
                       beta=0.1, )
        
        self.data_sampler = DataSampler(dataloader=train_ds, 
                                        model=self.expert, 
                                        save_dir="/fs/scratch/PAS2301/kumar1109/NISQA_Corpus", 
                                        K=15, 
                                        num_samples=1)
        self.DPO = DPO(sft_model=self.expert,
                       model=self.actor,   
                       gpu_id=gpu_id, 
                       beta=0.1)
        
    def train(self):
        print("Start training...")
        train_dl = self.data_sampler.generate_triplets()

        for epoch in range(self.args.epochs):
            for step, batch in enumerate(train_dl):
                x, ypos, yneg = batch
                
                #Get DPO loss
                loss = self.DPO.forward_step(x, ypos, yneg)
                loss = loss / self.accum_grad
                loss.backward()

                print(f"STEP:{step}|DPO_LOSS:{loss}")
        
                #Update network
                if not (torch.isnan(loss).any() or torch.isinf(loss).any()) and ((step+1) % self.accum_grad == 0):
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()


if __name__ == '__main__':

    train_ds, test_ds = load_data("/users/PAS2301/kumar1109/NISQA_Corpus", 
                                  4, 1, 
                                  32000, gpu = False)
    
    class Args:
        def __init__(self, batchsize, ckpt, n_fft, hop, gpu_id, init_lr, epochs):
            self.batchsize = batchsize
            self.ckpt = ckpt
            self.n_fft = n_fft
            self.hop = hop
            self.gpu_id = gpu_id
            self.epochs = epochs
            self.init_lr = init_lr

    args = Args(4, "/users/PAS2301/kumar1109/CMGAN/src/best_ckpt/ckpt", 400, 100, 0, 1e-05, 5)
    
    trainer = DPOTrainer(train_ds, test_ds, args=args, gpu_id=0)
    trainer.train()

 
            
