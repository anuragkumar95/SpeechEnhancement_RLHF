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
    parser.add_argument("-clean", "--clean_dir", type=str, required=True,
                        help="Root directory to clean audio files.")
    parser.add_argument("-rank", "--rank_dir", type=str, required=False,
                        help="Root directory to rank files.")
    parser.add_argument("--exp", type=str, required=False, default='default', help="Experiment name.")
    parser.add_argument("--suffix", type=str, required=False, default='', help="Experiment suffix name.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for checkpoints. Will create one if doesn't exist")
    parser.add_argument("-pt", "--ckpt", type=str, required=True, default=None,
                        help="Path to saved cmgan checkpoint.")
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

        self.actor = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1,
                            distribution="Normal", 
                            gpu_id=gpu_id)
        
        
        cmgan_expert_checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        self.actor.load_state_dict(cmgan_expert_checkpoint['generator_state_dict']) 
        #Freeze the policy
        self.actor = freeze_layers(self.actor, 'all')
        print(f"Loaded checkpoint stored at {args.ckpt}. Resuming training...") 
        del cmgan_expert_checkpoint 

        self.reward_model = RewardModel(policy=self.actor)

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
        _, x_1, x_2, labels = batch

        if self.gpu_id is not None:
            x_1 = x_1.to(self.gpu_id)
            x_2 = x_2.to(self.gpu_id)
            labels = labels.to(self.gpu_id)

        loss, score = self.reward_model(pos=x_1, neg=x_2)
        y_preds = (score < 0.5).float()
        labels = torch.argmax(labels, dim=-1)
        print(f"PREDS:{y_preds.reshape(-1)}")
        print(f"LABELS:{labels}")
        acc = self.accuracy(y_preds, labels.float())

        return loss, acc

    def train_one_epoch(self, epoch, train_ds, test_ds):
        #Run training
        num_batches = len(train_ds)
        train_loss = 0
        train_acc = 0
        batch_loss = 0
        batch_acc = 0
        run_val_every = 1000
        for i, batch in enumerate(train_ds):   
            self.reward_model.train()
            #clean, noisy, enh, _ = batch
            if len(batch) == 4:
                mini_batch_pairs = [(0, 1), (2, 1), (0, 2)]
            elif len(batch) == 3:
                mini_batch_pairs = [(0, 1)]
            for pair in mini_batch_pairs:
                pos, neg = batch[pair[0]], batch[pair[1]]
                labels = torch.tensor([1.0, 0.0]).repeat(self.args.batchsize, 1)
                mini_batch = (pos, neg, labels)
                mini_batch = preprocess_batch(mini_batch, gpu_id=self.gpu_id)
                try:  
                    loss, acc = self.forward_step(mini_batch)
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
                print(f"Epoch:{epoch} | Step:{i+1} | Loss: {batch_loss / len(mini_batch_pairs)} | Acc: {batch_acc / len(mini_batch_pairs)}")

            if (i+1) % run_val_every == 0:
                #Run validation
                self.reward_model.eval()
                num_batches = len(test_ds)
                val_loss = 0
                val_acc = 0
                with torch.no_grad():
                    for i, batch in enumerate(test_ds):
                        if len(batch) == 4:
                            mini_batch_pairs = [(0, 1), (2, 1), (0, 2)]
                        elif len(batch) == 3:
                            mini_batch_pairs = [(0, 1)]
                        for pair in mini_batch_pairs:
                            pos, neg = batch[pair[0]], batch[pair[1]]
                            labels = torch.tensor([1.0, 0.0]).repeat(self.args.batchsize, 1)
                            mini_batch = (pos, neg, labels)
                            mini_batch = preprocess_batch(mini_batch, gpu_id=self.gpu_id)
                            try:  
                                batch_loss, batch_acc = self.forward_step(mini_batch)
                            except Exception as e:
                                print(traceback.format_exc())
                                continue

                            if torch.isnan(batch_loss).any() or torch.isinf(batch_loss).any():
                                continue

                            val_loss += batch_loss.item()
                            val_acc += batch_acc
                    
                val_loss = val_loss / (num_batches * len(mini_batch_pairs))
                val_acc = val_acc / (num_batches * len(mini_batch_pairs))

                wandb.log({
                    'step': i+1,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })
                print(f"Epoch:{epoch} | Step:{i+1} | Val_Loss: {val_loss} | Val_Acc: {val_acc}")
    
            wandb.log({
                "step": i+1,
                "batch_loss":batch_loss.item(),
                "batch_acc":batch_acc / len(mini_batch_pairs),
            })
            batch_acc = 0
            batch_loss = 0
        
        train_loss = train_loss * self.ACCUM_GRAD / (num_batches * len(mini_batch_pairs))
        train_acc = train_acc * self.ACCUM_GRAD / (num_batches * len(mini_batch_pairs))

        return val_loss, val_acc, train_loss, train_acc

    def train(self, train_ds, test_ds):
        best_val = 99999999
        best_acc = 0
        print("Start training...")
        for epoch in range(self.args.epochs):
            val_loss, val_acc, tr_loss, tr_acc = self.train_one_epoch(epoch+1, train_ds, test_ds)
            #TODO:Log these in wandb
            wandb.log({
                "Epoch":epoch+1,
                "Val_loss":val_loss,
                "Train_loss":tr_loss,
                "Val_acc":val_acc,
                "Train_acc":tr_acc
            })
            
            if val_loss <= best_val or val_acc >= best_acc:
                best_val = val_loss
                best_acc = val_acc
                #TODO:Logic for savecheckpoint
                if self.gpu_id == 0:
                    checkpoint_prefix = f"{args.exp}_valLoss_{val_loss}_val_acc_{val_acc}_epoch_{epoch}.pt"
                    path = os.path.join(args.output, f"{args.exp}_{args.suffix}", checkpoint_prefix)
                    torch.save(self.reward_model.state_dict(), path)
                #TODO:May need a LR scheduler as well

def main(args):

    if args.gpu:
        trainer = Trainer(args, 0)
    else:
        trainer = Trainer(args, None)
    """
    speech_env = SpeechEnhancementAgent(n_fft=400,
                                        hop=100,
                                        gpu_id=None,
                                        args=None,
                                        reward_model=None)
    
    #enhance_model = copy.deepcopy(trainer.actor)
    #enhance_model = None
    
    train_dataset = PreferenceDataset(jnd_root=args.jndroot, 
                                      vctk_root=args.vctkroot, 
                                      set="train", 
                                      comp=args.comp,
                                      train_split=0.8, 
                                      resample=16000,
                                      enhance_model=enhance_model,
                                      env=speech_env,
                                      gpu_id=0, 
                                      cutlen=40000)
    
    test_dataset = PreferenceDataset(jnd_root=args.jndroot, 
                                     vctk_root=args.vctkroot, 
                                     set="test", 
                                     comp=args.comp,
                                     train_split=0.8, 
                                     resample=16000,
                                     enhance_model=enhance_model,
                                     env=speech_env,
                                     gpu_id=0,  
                                     cutlen=40000)
    """
    train_dataset = HumanAlignedDataset(mixture_dir=args.mix_dir,
                                        clean_dir=args.clean_dir, 
                                        rank=os.path.join(args.rank_dir, 'train.ranks'),  
                                        cutlen=40000)
    
    test_dataset = HumanAlignedDataset(mixture_dir=args.mix_dir,
                                      clean_dir=args.clean_dir, 
                                      rank=os.path.join(args.rank_dir, 'test.ranks'),  
                                      cutlen=40000)
    
    print(f"TRAIN:{len(train_dataset)} TEST:{len(test_dataset)}")

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
    
    
    trainer.train(train_dataloader, test_dataloader)
   

if __name__=='__main__':
    args = ARGS().parse_args()
    if len(args.suffix) > 0: 
        output = f"{args.output}/{args.exp}_{args.suffix}"
    else:
        output = f"{args.output}/{args.exp}"
    os.makedirs(output, exist_ok=True)
    main(args)