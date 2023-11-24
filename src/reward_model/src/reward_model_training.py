# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
Created on 23rd Nov, 2023
"""

"""
* TAKE THE DISCRIMINATOR WITH PRETRAINED WEIGHTS AS THE REWARD MODEL.
* PREPARE DATASET USING JND DATASET IN THE FOLLOWING WAY:-
    1. IF DIFFERENT, THEN CLEAN IS LABELED AS 1
    2. IF SAME, THEN IGNORE?
* WRITE A TRAIN SCRIPT USING THE LOSS OBJECTIVE DEFINED IN THE 
  ORIGINAL RLHF PAPER.
"""

from models.reward_model import  RewardModel, power_compress, power_uncompress
import os
from dataset.dataset import load_data
import torch.nn.functional as F
import torch
import torch.nn as nn
import argparse
import wandb
import psutil
import numpy as np
import traceback

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


wandb.login()

def ARGS():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True,
                        help="Root directory to JND Dataset.")
    parser.add_argument("-c", "--comp", type=str, required=True,
                        help="Root directory to JND Dataset comparision lists.")
    parser.add_argument("--exp", type=str, required=False, default='default', help="Experiment name.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for checkpoints. Will create one if doesn't exist")
    parser.add_argument("-pt", "--ckpt", type=str, required=False, default=None,
                        help="Path to saved cmgan checkpoint for resuming training.")
    parser.add_argument("--epochs", type=int, required=False, default=5,
                        help="No. of epochs to be trained.")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--gpu", action='store_true',
                        help="Set this flag for single gpu training.")
    parser.add_argument("--parallel", action='store_true',
                        help="Set this flag for parallel gpu training.")

    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--cut_len", type=int, default=40000, help="cut length")
    return parser
    
class Trainer:
    def __init__(self, train_ds, test_ds, args, gpu_id):
        self.model = RewardModel(ndf=32, in_channel=2)
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        if gpu_id is not None:
            self.model = self.model.to(gpu_id)

        self.gpu_id = gpu_id
        self.args = args
        wandb.init(project=args.exp)

    def get_specs(self, clean, noisy):
        """
        Create spectrograms from input waveform.
        ARGS:
            clean : clean waveform (batch * cut_len)
            noisy : noisy waveform (batch * cut_len)

        Return
            noisy_spec : (b * 2 * f * t) noisy spectrogram
            clean_spec : (b * 2 * f * t) clean spectrogram
        """
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )
        
        win = torch.hamming_window(self.n_fft)
        if self.gpu_id is not None:
            win = win.to(self.gpu_id)

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=win,
            onesided=True,
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=win,
            onesided=True,
        )

        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
        noisy_real = noisy_spec[:, 0, :, :].unsqueeze(1)
        noisy_imag = noisy_spec[:, 1, :, :].unsqueeze(1)
        noisy_mag = torch.sqrt(noisy_real**2 + noisy_imag**2)

        return noisy_mag, clean_mag
    
    def save(self, path, state_dict):
        torch.save(state_dict, path)
    
    def load(self, path):
        state_dict = torch.load(path)
        return state_dict
    
    def forward_step(self, batch):
        wav_in, wav_out, labels = batch
        if self.gpu_id is not None:
            wav_in = wav_in.to(self.gpu_id)
            wav_out = wav_out.to(self.gpu_id)
            labels = labels.to(self.gpu_id)

        class_probs = self.model(wav_in, wav_out)
        loss = self.criterion(class_probs, labels)
        return loss


    def train_one_epoch(self, epoch):
        #Run train loop
        epoch_loss = 0
        num_batches = len(self.train_ds)
        self.model.train()
        for i, batch in enumerate(self.train_ds):
            wav_in, wav_out, labels = batch
            wav_in, wav_out = self.get_specs(wav_in, wav_out)
            batch = (wav_in, wav_out, labels)
            
            batch_loss = self.forward_step(batch)
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            wandb.log({
                'epoch':epoch+1, 
                'step':i+1,
                'step_loss':batch_loss.detach()
            })

            print(f"EPOCH: {epoch+1} | STEP: {i+1} | LOSS: {batch_loss}")

            epoch_loss += batch_loss.detach()
        epoch_loss = epoch_loss / num_batches

        #Run validation
        val_loss = 0
        num_batches = len(self.val_ds)
        self.model.eval()
        for i, batch in enumerate(self.val_ds):
            wav_in, wav_out, labels = batch
            wav_in, wav_out = self.get_specs(wav_in, wav_out)
            batch = (wav_in, wav_out, labels)
            
            batch_loss = self.forward_step(batch)
            
            val_loss += batch_loss.detach()
        val_loss = val_loss / num_batches
        wandb.log({
                'epoch':epoch+1, 
                'val_loss':val_loss,
                'train_loss':epoch_loss, 
            })

        print(f"EPOCH: {epoch+1} | TRAIN_LOSS: {epoch_loss} | VAL_LOSS: {val_loss}")
        return val_loss

    def train(self):
        best_val_loss = 9999999999
        for epoch in range(self.args.epochs):
            val_loss = self.train_one_epoch(epoch)
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.args.output, self.args.exp, f"best_checkpoint_{val_loss}_epoch_{epoch+1}.pt")
                if args.parallel:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                self.save(save_path, state_dict)
                                
def main(args):
    if args.parallel:
        train_ds, test_ds = load_data(root=args.root, 
                                      path_root=args.comp,
                                      batch_size=args.batchsize, 
                                      n_cpu=1,
                                      split_ratio=0.85, 
                                      cut_len=args.cut_len,
                                      resample=True,
                                      parallel=True)
    else:
        train_ds, test_ds = load_data(root=args.root, 
                                      path_root=args.comp,
                                      batch_size=args.batchsize, 
                                      n_cpu=1,
                                      split_ratio=0.85, 
                                      cut_len=args.cut_len,
                                      resample=True,
                                      parallel=False)

    if args.gpu:
        trainer = Trainer(train_ds, test_ds, args, 0)
    else:
        trainer = Trainer(train_ds, test_ds, args, None)
    trainer.train()
   

if __name__=='__main__':
    args = ARGS().parse_args()
    output = f"{args.output}/{args.exp}"
    os.makedirs(output, exist_ok=True)
    main(args)