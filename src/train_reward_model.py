# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.actor import TSCNet
from model.reward_model import RewardModel
from model.critic import QNet
#from model.cmgan import TSCNet
from RLHF import REINFORCE
from reward_model.src.dataset.dataset import HumanAlignedDataset
from torch.utils.data import DataLoader
import copy

import os
import torch.nn.functional as F
import torch
from utils import preprocess_batch, power_compress, power_uncompress, batch_pesq, copy_weights, freeze_layers, original_pesq
import logging
from torchinfo import summary
import argparse
import wandb
import psutil
import numpy as np
import traceback

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from speech_enh_env import SpeechEnhancementAgent

def ARGS():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mix", "--mix_dir", type=str, required=True,
                        help="Root directory to audio mixtures.")
    parser.add_argument("-rank", "--rank_dir", type=str, required=False,
                        help="Root directory to rank files.")
    parser.add_argument("-vr", "--vctk_root", type=str, required=False,
                        help="Root directory to voicebank dataset.")
    parser.add_argument("--exp", type=str, required=False, default='default', help="Experiment name.")
    parser.add_argument("--suffix", type=str, required=False, default='', help="Experiment suffix name.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for checkpoints. Will create one if doesn't exist")
    parser.add_argument("--epochs", type=int, required=False, default=5,
                        help="No. of epochs to be trained.")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--accum_grad", type=int, required=False, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--gpu", action='store_true',
                        help="Set this flag for single gpu training.")
    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    
    return parser

wandb.login()


class Trainer:
    """
    Reward model training.
    """
    def __init__(self, args, gpu_id):
        self.n_fft = 400
        self.hop = 100
    
        self.ACCUM_GRAD = args.accum_grad

        self.reward_model = RewardModel(in_channels=2)

        self.a_optimizer = torch.optim.AdamW(
            filter(lambda layer:layer.requires_grad,self.reward_model.parameters()), lr=args.init_lr
        )

        if gpu_id is not None:
            self.reward_model = self.reward_model.to(gpu_id)

        self.gpu_id = gpu_id
        self.args = args
        wandb.init(project=args.exp)

    def save(self, path, state_dict):
        torch.save(state_dict, path)
    
    def load(self, path, device):
        if device == 'cpu':
            dev = torch.device('cpu')
        else:
            dev = torch.device(f'cuda:{device}')
        state_dict = torch.load(path, map_location=dev)
        return state_dict
    
    def accuracy(self, y_pred, y_true):
        score = (y_pred == y_true).float()
        return score.mean()

    def forward_step(self, batch):
        _, x_1, x_2, ref, labels = batch

        if self.gpu_id is not None:
            x_1 = x_1.to(self.gpu_id)
            x_2 = x_2.to(self.gpu_id)
            ref = ref.to(self.gpu_id)
            labels = labels.to(self.gpu_id)

        labels = torch.argmax(labels, dim=-1)
        loss, score, probs = self.reward_model(pos=x_1, neg=x_2, ref=ref, labels=labels)
        if probs is None:
            y_preds = (score < 0.5).float()
        else:
            y_preds = torch.argmax(probs, dim=-1)
        
        print(f"PREDS:{y_preds.reshape(-1)}")
        print(f"LABELS:{labels}")
        acc = self.accuracy(y_preds, labels.float())

        return loss, acc
    
    def run_validation(self, test_ds):
        #Run validation
        self.reward_model.eval()
        num_batches = len(test_ds)
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for i, batch in enumerate(test_ds):
                
                pos, neg, ref, labels, _ = batch
                batch = (pos, neg, labels)
                batch = preprocess_batch(batch, ref=ref, gpu_id=self.gpu_id)
                try:  
                    batch_loss, batch_acc = self.forward_step(batch)

                except Exception as e:
                    print(traceback.format_exc())
                    continue

                if torch.isnan(batch_loss).any() or torch.isinf(batch_loss).any():
                    continue

                val_loss += batch_loss.item()
                val_acc += batch_acc
            
        val_loss = val_loss / num_batches
        val_acc = val_acc / num_batches
        self.reward_model.train()
        return val_loss, val_acc 
        
    def train_one_epoch(self, epoch, train_ds, test_ds, best_val_loss, best_val_acc):
        #Run training
        num_batches = len(train_ds)
        train_loss = 0
        train_acc = 0
        batch_loss = 0
        batch_acc = 0
    
        for i, batch in enumerate(train_ds):   
            pos, neg, labels, _ = batch
            batch = (pos, neg, labels)
            batch = preprocess_batch(batch, gpu_id=self.gpu_id)
            try:  
                loss, acc = self.forward_step(batch)
            except Exception as e:
                print(traceback.format_exc())
                continue
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue
            
            batch_loss += loss / self.ACCUM_GRAD
            batch_acc += acc

            self.a_optimizer.zero_grad()
            batch_loss.backward()

            if (i+1) % self.ACCUM_GRAD == 0 or i+1 == num_batches:
                #torch.nn.utils.clip_grad_value_(self.actor.parameters(), 5.0)
                self.a_optimizer.step()

                train_loss += batch_loss.item()
                train_acc += batch_acc
                print(f"Epoch:{epoch} | Step:{i+1} | Loss: {batch_loss} | Acc: {batch_acc}")

            if (i+1) % 3000  == 0:
                #Run validation every 1000 steps
                val_loss, val_acc = self.run_validation(test_ds)
                wandb.log({
                    'STEP': (num_batches * (epoch-1)) + i + 1,
                    'val_acc':val_acc,
                    'val_loss':val_loss
                })
                print(f"Epoch:{epoch} | Step:{i+1} | Val_Loss: {val_loss} | Val_Acc: {val_acc}")

                if val_loss < best_val_loss or best_val_acc < val_acc:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                    if self.gpu_id == 0:
                        checkpoint_prefix = f"{args.exp}_valLoss_{val_loss}_val_acc_{val_acc}_epoch_{epoch}_step_{i+1}.pt"
                        path = os.path.join(args.output, f"{args.exp}_{args.suffix}", checkpoint_prefix)
                        torch.save(self.reward_model.state_dict(), path)

            wandb.log({
                "step": i+1,
                "batch_loss":batch_loss.item(),
                "batch_acc":batch_acc,
            })
            batch_acc = 0
            batch_loss = 0
        
        train_loss = train_loss * self.ACCUM_GRAD / num_batches
        train_acc = train_acc * self.ACCUM_GRAD / num_batches

        return best_val_loss, best_val_acc

    def train(self, train_ds, test_ds):
        print("Start training...")
        best_val_loss = 99999
        best_val_acc = 0
        for epoch in range(self.args.epochs):
            best_val_loss, best_val_acc = self.train_one_epoch(epoch+1, train_ds, test_ds, best_val_loss, best_val_acc)

            #Run last validation
            val_loss, val_acc = self.run_validation(test_ds)
            wandb.log({
                'epoch':epoch+1, 
                'val_acc':val_acc,
                'val_loss':val_loss
            })
            print(f"Epoch:{epoch} | Val_Loss: {val_loss} | Val_Acc: {val_acc}")

            if val_loss < best_val_loss or best_val_acc < val_acc:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
               
                if self.gpu_id == 0:
                    checkpoint_prefix = f"{args.exp}_valLoss_{val_loss}_val_acc_{val_acc}_epoch_{epoch}.pt"
                    path = os.path.join(args.output, f"{args.exp}_{args.suffix}", checkpoint_prefix)
                    torch.save(self.reward_model.state_dict(), path)
        

def main(args):

    if args.gpu:
        trainer = Trainer(args, 0)
    else:
        trainer = Trainer(args, None)
 
    train_dataset = HumanAlignedDataset(mixture_dir=os.path.join(args.mix_dir, 'train'),
                                        noisy_dir=os.path.join(args.vctk_root, 'train', 'noisy'),
                                        rank=os.path.join(args.rank_dir, 'train.ranks'),  
                                        cutlen=40000)
    
    test_dataset = HumanAlignedDataset(mixture_dir=os.path.join(args.mix_dir, 'test'),
                                       noisy_dir=os.path.join(args.vctk_root, 'test', 'noisy'),
                                       rank=os.path.join(args.rank_dir, 'test.ranks'),  
                                       cutlen=40000)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=1,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batchsize,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=1,
    )
    
    print(f"TRAIN:{len(train_dataloader)} TEST:{len(test_dataloader)}")
    
    trainer.train(train_dataloader, test_dataloader)
   

if __name__=='__main__':
    args = ARGS().parse_args()
    if len(args.suffix) > 0: 
        output = f"{args.output}/{args.exp}_{args.suffix}"
    else:
        output = f"{args.output}/{args.exp}"
    os.makedirs(output, exist_ok=True)
    main(args)