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
from utils import power_compress, batch_pesq, copy_weights, freeze_layers, 

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

class DPO:
    def __init__(self,
                 sft_model,
                 model,   
                 gpu_id=None, 
                 beta=0.2,
                 **params):
        
        self.ref_model = sft_model
        self.ref_model.eval()
        self.model = model
        self.gpu_id = gpu_id 
        self.std = 0.01
        self.beta = beta
        self.n_fft = params.get("n_fft")
        self.hop = params.get("hop")


    def get_logprob(self, mu, x):
        std = (torch.ones(mu.shape) * self.std).to(self.gpu_id)
        N = Normal(mu, std)
        x_logprob = N.log_prob(x)
        return x_logprob 

    def dpo_loss(self, x, ypos, yneg):
        ref_mu = self.ref_model(x)
        y_mu = self.model(x)

        ref_pos_logprob = self.get_logprob(ref_mu, ypos)
        ref_neg_logprob = self.get_logprob(ref_mu, yneg)

        y_pos_logprob = self.get_logprob(y_mu, ypos)
        y_neg_logprob = self.get_logprob(y_mu, yneg)

        scores = self.beta * ((y_pos_logprob - ref_pos_logprob) - (y_neg_logprob - ref_neg_logprob)) 
        log_scores = torch.log(F.sigmoid(scores)).mean()

        return -log_scores

    
    def spec(self, noisy, ypos, yneg):
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
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        ypos_spec = torch.stft(
            ypos,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        yneg_spec = torch.stft(
            yneg,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )

        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        ypos_spec = power_compress(ypos_spec)
        yneg_spec = power_compress(yneg_spec)
        
        return noisy_spec, ypos_spec, yneg_spec


    def forward_step(self, x, ypos, yneg):
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
            if args.model == 'cmgan':
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

        self.DPO = DPO(loader=train_ds,
                       sft_model=self.expert,
                       model=self.actor,   
                       gpu_id=gpu_id, 
                       beta=0.1, 
                       params={'n_fft':400, 'hop':100})
        
        self.data_sampler = DataSampler(dataloader=train_ds, 
                                        model=self.expert, 
                                        save_dir="/fs/scratch/PAS2301/kumar1109/NISQA_Corpus", 
                                        K=15, 
                                        num_samples=10)
        self.DPO = DPO(loader=train_ds,
                       sft_model=self.expert,
                       model=self.actor,   
                       gpu_id=gpu_id, 
                       beta=0.1)
        
    def train(self):

        train_dl = self.data_sampler.generate_triplets()

        for epoch in range(self.args.epochs):
            for step, batch in enumerate(train_dl):
                x, ypos, yneg = batch
                
                #Get DPO loss
                loss = self.DPO.forward_step(x, ypos, yneg)
                loss = loss / self.accum_grad
                loss.backward()
        
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
        def __init__(self, batchsize, ckpt, n_fft, hop, gpu_id):
            self.batchsize = batchsize
            self.ckpt = ckpt
            self.n_fft = n_fft
            self.hop = hop
            self.gpu_id = gpu_id

    args = Args(4, "/users/PAS2301/kumar1109/CMGAN/src/best_ckpt/ckpt", 400, 100, 0)
    
    trainer = DPOTrainer(train_ds, test_ds, args=args, gpu_id=0)

 
            
