# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.actor import TSCNet, RewardModel
from model.critic import QNet
#from model.cmgan import TSCNet
from RLHF import REINFORCE
from reward_model.src.dataset.dataset import load_data
#import cdpam

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
    parser.add_argument("-r", "--root", type=str, required=True,
                        help="Root directory to JND.")
    parser.add_argument("-c", "--comp", type=str, required=False,
                        help="Root directory to JND Dataset comparision lists.")
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

    parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
    
    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                    "and dereverberation")
    return parser

wandb.login()


class Trainer:
    """
    Reward model training.
    """
    def __init__(self, train_ds, test_ds, args, gpu_id):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.ACCUM_GRAD = args.accum_grad

        self.actor = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1,
                            distribution=True, 
                            gpu_id=gpu_id)
        
        
        cmgan_expert_checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        self.actor.load_state_dict(cmgan_expert_checkpoint['generator_state_dict']) 
        #Freeze the policy
        self.actor = freeze_layers(self.actor, 'all')
        print(f"Loaded checkpoint stored at {args.ckpt}. Resuming training...") 
        del cmgan_expert_checkpoint 

        self.reward_model = RewardModel(policy=self.actor)
        #self.cdpam = cdpam.CDPAM(dev=gpu_id)

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

    def forward_step(self, batch, dist=None):
        _, x_1, x_2, labels = batch

        if self.gpu_id is not None:
            x_1 = x_1.to(self.gpu_id)
            x_2 = x_2.to(self.gpu_id)
            labels = labels.to(self.gpu_id)

        probs = self.reward_model(x_1, x_2, dist)
        loss = F.cross_entropy(probs, labels)

        y_preds = torch.argmax(probs, dim=-1)
        labels = torch.argmax(labels, dim=-1)
        print(f"PREDS:{y_preds}")
        print(f"LABELS:{labels}")
        acc = self.accuracy(y_preds.float(), labels.float())

        return loss, acc

    def train_one_epoch(self, epoch):
        #Run training
        self.reward_model.train()
        num_batches = len(self.train_ds)
        train_loss = 0
        train_acc = 0
        batch_loss = 0
        for i, batch in enumerate(self.train_ds):   
            ##Calculate cdpam distance
            #wav_ref, wav_inp, _ = batch
            #cdpam_dist = self.cdpam.forward(wav_ref, wav_inp)

            #Each minibatch is an episode
            batch = preprocess_batch(batch, gpu_id=self.gpu_id)
            try:  
                loss, batch_acc = self.forward_step(batch)
            except Exception as e:
                print(traceback.format_exc())
                continue

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue
            
            batch_loss += loss / self.ACCUM_GRAD

            self.a_optimizer.zero_grad()
            batch_loss.backward()

            if (i+1) % self.ACCUM_GRAD == 0 or i+1 == num_batches:
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), 5.0)
                self.a_optimizer.step()

                train_loss += batch_loss.item()
                train_acc += batch_acc
                print(f"Epoch:{epoch} | Step:{i+1} | Loss: {batch_loss} | Acc: {batch_acc}")

                batch_loss = 0
            
            wandb.log({
                "step": i+1,
                "batch_loss":loss.item(),
                "batch_acc":batch_acc
            })
            

        train_loss = train_loss * self.ACCUM_GRAD / num_batches
        train_acc = train_acc * self.ACCUM_GRAD / num_batches

        #Run validation
        self.reward_model.eval()
        num_batches = len(self.test_ds)
        val_loss = 0
        val_acc = 0
        for i, batch in enumerate(self.test_ds):
            
            #Each minibatch is an episode
            batch = preprocess_batch(batch, gpu_id=self.gpu_id)
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

        print(f"Epoch:{epoch} | Val_Loss: {val_loss} | Val_Acc: {val_acc}")

        return val_loss, val_acc, train_loss, train_acc

    def train(self):
        best_val = 99999999
        print("Start training...")
        for epoch in range(self.args.epochs):
            val_loss, val_acc, tr_loss, tr_acc = self.train_one_epoch(epoch+1)
            #TODO:Log these in wandb
            wandb.log({
                "Epoch":epoch+1,
                "Val_loss":val_loss,
                "Train_loss":tr_loss,
                "Val_acc":val_acc,
                "Train_acc":tr_acc
            })
            
            if val_loss >= best_val:
                best_val = val_loss
                #TODO:Logic for savecheckpoint
                if self.gpu_id == 0:
                    checkpoint_prefix = f"{args.exp}_valLoss_{val_loss}_epoch_{epoch}.pt"
                    path = os.path.join(args.output, args.exp, checkpoint_prefix)
                    torch.save(self.reward_model.state_dict(), path)
                #TODO:May need a LR scheduler as well

def main(args):

    if args.root.endswith('.npy'):
        train_ds, test_ds = load_data(data=args.root, 
                                        batch_size=args.batchsize, 
                                        n_cpu=1,
                                        split_ratio=0.8, 
                                        cut_len=args.cut_len,
                                        resample=True,
                                        parallel=False)
    else:
        train_ds, test_ds = load_data(root=args.root, 
                                        path_root=args.comp,
                                        batch_size=args.batchsize, 
                                        n_cpu=1,
                                        type='linear',
                                        split_ratio=0.8, 
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
    if len(args.suffix) > 0: 
        output = f"{args.output}/{args.exp}_{args.suffix}"
    else:
        output = f"{args.output}/{args.exp}"
    os.makedirs(output, exist_ok=True)
    main(args)