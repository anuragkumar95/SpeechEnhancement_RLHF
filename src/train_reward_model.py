# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.CMGAN.actor import TSCNet
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
from utils import preprocess_batch, get_specs_1, freeze_layers
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
    parser.add_argument("-pt", "--reward_pt", type=str, required=False,
                        help="Path to SFT model checkpoint.")
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

        #policy = TSCNet(num_channel=64, 
        #                num_features=201,
        #                distribution=None, 
        #                gpu_id=gpu_id)
        
        #Load checkpoint and freeze layers
        #sft_checkpoint = torch.load(args.model_pt, map_location=torch.device('cpu'))
        #policy.load_state_dict(sft_checkpoint)
        #policy = freeze_layers(policy, 'all')
        
        #policy = policy.to(gpu_id)

        self.reward_model = RewardModel(in_channels=2)
        checkpoint = torch.load(args.reward_pt, map_location=torch.device('cpu'))
        self.reward_model.load_state_dict(checkpoint)
        self.reward_model = self.reward_model.to(gpu_id)
        #self.reward_model = self.reward_model.eval()

        self.a_optimizer = torch.optim.AdamW(
            filter(lambda layer:layer.requires_grad,self.reward_model.parameters()), lr=args.init_lr
        )

        if gpu_id is not None:
            self.reward_model = self.reward_model.to(gpu_id)

        self.gpu_id = gpu_id
        self.args = args
        wandb.init(project=args.exp, name=args.suffix)

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
        #x_1, x_2, inp = batch
        _, x_1, x_2, inp, _ = batch
        labels = torch.ones(x_1.shape[0]).reshape(-1)
        if self.gpu_id is not None:
            labels = labels.to(self.gpu_id)

        loss, score, probs = self.reward_model(x=inp, pos=x_1, neg=x_2)
        
        if probs is None:
            pos_score, neg_score = score
            y_preds = (pos_score > neg_score).float().reshape(-1)
        else:
            y_preds = torch.argmax(probs, dim=-1)
        
        print(f"PREDS:{y_preds}")
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
                pos, neg, inp= batch
                pos = pos.squeeze(0)
                neg = neg.squeeze(0)
                inp = inp.squeeze(0)
                batch = (pos, neg, inp)
                batch = preprocess_batch(batch, ref=inp, gpu_id=self.gpu_id)
                #if self.gpu_id is not None:
                #    pos = pos.to(self.gpu_id)
                #    neg = neg.to(self.gpu_id)
                #    inp = inp.to(self.gpu_id)
                
                #pos = get_specs_1(wav=pos, n_fft=400, hop=100, gpu_id=self.gpu_id)
                #neg = get_specs_1(wav=neg, n_fft=400, hop=100, gpu_id=self.gpu_id)
                #inp = get_specs_1(wav=inp, n_fft=400, hop=100, gpu_id=self.gpu_id)

                
                #batch = (pos, neg, inp)
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
            pos, neg, inp= batch
            pos = pos.squeeze(0)
            neg = neg.squeeze(0)
            inp = inp.squeeze(0)
            batch = (pos, neg, inp)
            batch = preprocess_batch(batch, ref=inp, gpu_id=self.gpu_id)
            #if self.gpu_id is not None:
            #    pos = pos.to(self.gpu_id)
            #    neg = neg.to(self.gpu_id)
            #    inp = inp.to(self.gpu_id)
            
            #pos = get_specs_1(wav=pos, n_fft=400, hop=100, gpu_id=self.gpu_id)
            #neg = get_specs_1(wav=neg, n_fft=400, hop=100, gpu_id=self.gpu_id)
            #inp = get_specs_1(wav=inp, n_fft=400, hop=100, gpu_id=self.gpu_id)

            
            #batch = (pos, neg, inp)
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

            if (i+1) % 2000  == 0:
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

            if (val_loss < best_val_loss) or (best_val_acc < val_acc):
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
 
    train_dataset = HumanAlignedDataset(mixture_dir=os.path.join(args.mix_dir, 'train', 'audios'),
                                        noisy_dir=os.path.join(args.vctk_root, 'train', 'noisy'),
                                        rank=os.path.join(args.rank_dir, 'train.pairs'), 
                                        mos_file=os.path.join(args.rank_dir, 'NISQA_results_train.csv'),
                                        batchsize=args.batchsize,
                                        cutlen=40000)
    
    test_dataset = HumanAlignedDataset(mixture_dir=os.path.join(args.mix_dir, 'test', 'audios'),
                                       noisy_dir=os.path.join(args.vctk_root, 'test', 'noisy'),
                                       rank=os.path.join(args.rank_dir, 'test.pairs'),
                                       mos_file=os.path.join(args.rank_dir, 'NISQA_results_test.csv'),  
                                       cutlen=40000)
    
    print(f"TRAIN:{len(train_dataset)}, TEST:{len(test_dataset)}")

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


    