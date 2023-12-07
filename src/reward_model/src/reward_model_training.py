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

from models.reward_model import JNDModel, power_compress
from utils import copy_weights, freeze_layers, ContrastiveLoss
import os
from dataset.dataset import load_data
import torch.nn.functional as F
import torch
import torch.nn as nn
import argparse
import wandb
import numpy as np
import traceback
from model_evaluation import Evaluation

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
                        help="Path to saved checkpoint for resuming training.")
    parser.add_argument("--disc_pt", type=str, required=False, default=None,
                        help="Path to the discriminator checkpoint to init reward model.")
    parser.add_argument("--epochs", type=int, required=False, default=5,
                        help="No. of epochs to be trained.")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--norm", type=str, required=False, default='sbn',
                        help="option, choose between 'ln(layernorm) / sbn(batchnorm)'")
    parser.add_argument("--enc", type=int, required=False, default=1,
                        help="encoding option, choose between 1 or 2")
    parser.add_argument("--heads", type=int, required=False, default=1,
                        help="No of attention heads")
    parser.add_argument("--loss", type=str, required=False, default='featureloss',
                        help="option, choose between featureloss/attentionloss")
    parser.add_argument("--gpu", action='store_true',
                        help="Set this flag for single gpu training.")
    parser.add_argument("--parallel", action='store_true',
                        help="Set this flag for parallel gpu training.")
    parser.add_argument("--suffix", type=str, default='', required=False,
                        help="Experiment suffix to be added.")

    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--cut_len", type=int, default=40000, help="cut length")
    return parser
    
class Trainer:
    def __init__(self, train_ds, test_ds, args, gpu_id):
        
        self.model = JNDModel(in_channels=2,
                              out_dim=2, 
                              n_layers=7, 
                              keep_prob=0.7, 
                              norm_type=args.norm, 
                              sum_till=7, 
                              n_heads=args.heads,
                              gpu_id=gpu_id,
                              enc_type=args.enc,
                              loss_type=args.loss)
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.start_epoch = 0


        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.contrastive_loss = ContrastiveLoss(reduction='mean', eps=0.0005)

        if gpu_id is not None:
            self.model = self.model.to(gpu_id)

        if args.ckpt is not None:
            state_dict = self.load(args.ckpt, gpu_id)
            self.start_epoch = state_dict['epoch'] - 1
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['opt_state_dict'])
            print(f"Loaded checkpoint stored at {args.ckpt} with val_acc {state_dict['val_acc']} at epoch {self.start_epoch}")

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

        noisy_spec = power_compress(noisy_spec)
        clean_spec = power_compress(clean_spec)

        return noisy_spec, clean_spec
    
    def save(self, path, state_dict):
        torch.save(state_dict, path)
    
    def load(self, path, device):
        if device == 'cpu':
            dev = torch.device('cpu')
        else:
            dev = torch.device(f'cuda:{device}')
        state_dict = torch.load(path, map_location=dev)
        return state_dict
    
    def forward_step(self, batch):
        wav_in, wav_out, labels = batch
        #wav_in = wav_in.unsqueeze(1).unsqueeze(-1)
        #wav_out = wav_out.unsqueeze(1).unsqueeze(-1)
        class_probs, distances = self.model(wav_in, wav_out)
        loss_ce = self.criterion(class_probs, labels)
        loss_contrastive = self.contrastive_loss(distances, labels)
        loss = loss_ce + loss_contrastive
        return loss, class_probs


    def train_one_epoch(self, epoch):
        #Run train loop
        epoch_loss = 0
        epoch_acc = 0
        num_batches = len(self.train_ds)
        self.model.train()
        for i, batch in enumerate(self.train_ds):
            wav_in, wav_out, labels = batch
            if wav_in.shape[0] <= 1:
                continue
            if self.gpu_id is not None:
                wav_in = wav_in.to(self.gpu_id)
                wav_out = wav_out.to(self.gpu_id)
                labels = labels.to(self.gpu_id)

            wav_in, wav_out = self.get_specs(wav_in, wav_out)
            batch = (wav_in, wav_out, labels)
            
            batch_loss, probs = self.forward_step(batch)
            y_preds = torch.argmax(probs, dim=-1)
            labels = torch.argmax(labels, dim=-1)
            print(f"PREDS:{y_preds}")
            print(f"LABELS:{labels}")
            acc = self.accuracy(y_preds.float(), labels.float())

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            wandb.log({
                'epoch':epoch+1, 
                'step':i+1,
                'step_loss':batch_loss.detach(),
                'step_acc':acc.detach()
            })

            print(f"EPOCH: {epoch+1} | STEP: {i+1} | LOSS: {batch_loss} | ACC :{acc}")

            epoch_loss += batch_loss.detach()
            epoch_acc += acc.detach()
        epoch_loss = epoch_loss / num_batches
        epoch_acc = epoch_acc / num_batches

        #Run validation
        val_loss = 0
        val_acc = 0
        num_batches = len(self.test_ds)
        self.model.eval()
        with torch.no_grad():
          for i, batch in enumerate(self.test_ds):
              wav_in, wav_out, labels = batch
              if wav_in.shape[0] <= 1:
                continue
              if self.gpu_id is not None:
                  wav_in = wav_in.to(self.gpu_id)
                  wav_out = wav_out.to(self.gpu_id)
                  labels = labels.to(self.gpu_id)
              wav_in, wav_out = self.get_specs(wav_in, wav_out)
              
              batch = (wav_in, wav_out, labels)
              batch_loss, probs = self.forward_step(batch)
              y_preds = torch.argmax(probs, dim=-1)
              labels = torch.argmax(labels, dim=-1)
              print(f"PREDS:{y_preds}")
              print(f"LABELS:{labels}")
              acc = self.accuracy(y_preds.float(), labels.float())
              print(f"ACC:{acc}")
              val_loss += batch_loss.detach()
              val_acc += acc.detach()
        val_loss = val_loss / num_batches
        val_acc = val_acc / num_batches
        wandb.log({
                'epoch':epoch+1, 
                'val_loss':val_loss,
                'train_loss':epoch_loss, 
                'val_acc':val_acc,
                'train_acc':epoch_acc,
            })

        print(f"EPOCH: {epoch+1} | TRAIN_LOSS: {epoch_loss} | VAL_LOSS: {val_loss} | TRAIN_ACC: {epoch_acc} | VAL_ACC: {val_acc}")
        return val_loss, val_acc
    
    def accuracy(self, y_pred, y_true):
        score = (y_pred == y_true).float()
        return score.mean()

    def train(self):
        best_val_loss = 9999999999
        best_val_acc = 0
        for epoch in range(self.start_epoch, self.args.epochs):
            save_path = None
            val_loss, val_acc = self.train_one_epoch(epoch)
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.args.output, f"{self.args.exp}_{args.suffix}", f"best_checkpoint_{val_loss}_epoch_{epoch+1}_acc_{val_acc}.pt")
                if args.parallel:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                _dict_ = {
                    'model_state_dict':state_dict,
                    'opt_state_dict':self.optimizer.state_dict(),
                    'epoch':epoch+1,
                    'val_acc':val_acc,
                    'val_loss':val_loss
                }
                self.save(save_path, _dict_)
               
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(self.args.output, f"{self.args.exp}_{args.suffix}", f"best_checkpoint_{val_loss}_epoch_{epoch+1}_acc_{val_acc}.pt")
                if args.parallel:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                _dict_ = {
                    'model_state_dict':state_dict,
                    'opt_state_dict':self.optimizer.state_dict(),
                    'epoch':epoch+1,
                    'val_acc':val_acc,
                    'val_loss':val_loss
                }
                self.save(save_path, _dict_)
                              
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
    output = f"{args.output}/{args.exp}_{args.suffix}"
    os.makedirs(output, exist_ok=True)
    main(args)