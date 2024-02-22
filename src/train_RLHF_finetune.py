# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.actor import TSCNet, RewardModel
from model.critic import QNet
#from model.cmgan import TSCNet
from RLHF import REINFORCE, PPO


import os
from data.dataset import load_data
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

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True,
                        help="Root directory to Voicebank.")
    parser.add_argument("--exp", type=str, required=False, default='default', help="Experiment name.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for checkpoints. Will create one if doesn't exist")
    parser.add_argument("-pt", "--ckpt", type=str, required=True, default=None,
                        help="Path to saved checkpoint to fine-tune.")
    parser.add_argument("--reward_pt", type=str, required=False, default=None,
                        help="path to the reward model checkpoint.")
    parser.add_argument("--epochs", type=int, required=False, default=5,
                        help="No. of epochs to be trained.")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--accum_grad", type=int, required=False, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--gpu", action='store_true',
                        help="Set this flag for single gpu training.")
    parser.add_argument("--parallel", action='store_true',
                        help="Set this flag for parallel gpu training.")
    parser.add_argument("--out_dist", action='store_true',
                        help="If GAN learns a distribution.")
    parser.add_argument("--method", type=str, default='reinforce', required=False,
                        help="RL Algo to run. Choose between (reinforce/PPO)")
    
    parser.add_argument("--reward", type=int, help="Type of reward")
    parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
    
    parser.add_argument("--gamma", type=float, default=0.99, help="Reward discount factor")
    parser.add_argument("--tau", type=float, default=0.99, help="target critic soft update factor")
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                    "and dereverberation")
    return parser

wandb.login()


class Trainer:
    """
    Starting with reinforce algorithm.
    """
    def __init__(self, train_ds, test_ds, args, gpu_id, pretrain=False):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.ACCUM_GRAD = args.accum_grad

        self.actor = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1,
                            distribution=args.out_dist, 
                            gpu_id=gpu_id)
        
        self.expert = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1,
                            distribution=args.out_dist, 
                            gpu_id=gpu_id)
        
        self.reward_model = RewardModel(policy=self.actor)
        
        cmgan_expert_checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        self.actor.load_state_dict(cmgan_expert_checkpoint['generator_state_dict']) 
        self.expert.load_state_dict(cmgan_expert_checkpoint['generator_state_dict'])
        
        reward_checkpoint = torch.load(args.reward_pt, map_location=torch.device('cpu'))
        self.reward_model.load_state_dict(reward_checkpoint)
        
        #Freeze complex decoder and reward model
        self.actor = freeze_layers(self.actor, ['dense_encoder', 'TSCB_1', 'complex_decoder'])
        self.reward_model = freeze_layers(self.reward_model, 'all')
        self.reward_model.eval()

        #Set expert to eval and freeze all layers.
        self.expert = freeze_layers(self.expert, 'all')
        self.expert.eval()

        print(f"Loaded checkpoint stored at {args.ckpt}. Resuming training...") 
        del cmgan_expert_checkpoint 
        del reward_checkpoint

        if args.method == 'reinforce':
            
            self.optimizer = torch.optim.AdamW(
                filter(lambda layer:layer.requires_grad,self.actor.parameters()), lr=args.init_lr
            )

            self.trainer = REINFORCE(gpu_id=gpu_id, 
                                    beta = 1e-10 , 
                                    init_model=self.expert,
                                    discount=1.0,
                                    train_phase=True,
                                    reward_model=self.reward_model,
                                    env_params={'n_fft':400,
                                                'hop':100, 
                                                'args':args})
            
        if args.method == 'PPO':
            self.critic = QNet(ndf=16, in_channel=2, out_channel=1)
            params = list(self.actor.parameters()) + list(self.critic.parameters())
            self.optimizer = torch.optim.AdamW(
                filter(lambda layer:layer.requires_grad, params), lr=args.init_lr
            )

            self.trainer = PPO(init_model=self.expert, 
                               reward_model=self.reward_model, 
                               gpu_id=None, 
                               beta=0.01,
                               val_coef=0.5,
                               en_coef=0.01,
                               discount=1.0,
                               env_params={'n_fft':400,
                                            'hop':100, 
                                            'args':args})
        if gpu_id is not None:
            self.actor = self.actor.to(gpu_id)
            self.critic = self.critic.to(gpu_id)
            if args.parallel:
                self.actor = DDP(self.actor, device_ids=[gpu_id])
                self.critic = DDP(self.critic, device_ids=[gpu_id])

        self.gpu_id = gpu_id
        self.G = 0
        self.args = args
        
        wandb.init(project=args.exp)
    
    def run_validation(self, env, batch):
        """
        Runs a vlidation loop for a batch.
        Predict mask for each frame one at a time 
        and return pesq score of the enhances batch of 
        spectrograms.
        """
        #print("Running validation...")
        clean_aud, _, noisy, _ = batch
        inp = noisy.permute(0, 1, 3, 2)

        #Forward pass through actor to get the action(mask)
        action, _, _ = self.actor.get_action(inp)
        exp_action, _, _ = self.expert.get_action(inp)
        a_t = (action[0], exp_action[-1])
        
        #Apply action  to get the next state
        next_state = env.get_next_state(state=inp, 
                                        action=a_t)

        pesq, pesq_mask = batch_pesq(clean_aud.detach().cpu().numpy(), 
                                     next_state['est_audio'].detach().cpu().numpy())
        return (pesq*pesq_mask).mean()
    

    def train_one_epoch(self, epoch):
        #Run training
        self.actor.train()
        if self.args.method == 'PPO':
            self.critic.train()
        REWARDS = []
        num_batches = len(self.train_ds)
        train_ep_PESQ = 0
        self.trainer.t = 0
        for i, batch in enumerate(self.train_ds):   
           
            #Each minibatch is an episode
            batch = preprocess_batch(batch, gpu_id=self.gpu_id) 
            try:  
                loss, batch_reward, G = self.trainer.run_episode(batch, self.actor, self.optimizer)
            except Exception as e:
                print(traceback.format_exc())
                continue

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue
            
            train_ep_PESQ += original_pesq(batch_reward.item()) 

            wandb.log({
                "episode_cumulative_reward":batch_reward.item(),
                "trainPESQ":original_pesq(batch_reward.item()),
                "episode": (i+1) + ((epoch - 1) * num_batches),
                "G_t":G,
                "cumulative_G_t": G + self.G, 
                "loss":loss.item(),
                #"lr":self.lr_scheduler.get_last_lr()
            })

            self.G = G + self.G
            print(f"Epoch:{epoch} | Episode:{i+1} | Reward: {batch_reward}")
            REWARDS.append(batch_reward.item())

        train_ep_PESQ = train_ep_PESQ / num_batches
        
        #Run validation
        self.actor.eval()
        if self.args.method == 'PPO':
            self.critic.eval()
        pesq = 0
        v_step = 0
        for i, batch in enumerate(self.test_ds):
            
            #Preprocess batch
            batch = preprocess_batch(batch, gpu_id=self.gpu_id)
            
            #Run validation episode
            try:
                val_pesq_score = self.run_validation(self.trainer.env, batch)
            except Exception as e:
                print(traceback.format_exc())
                continue

            pesq += val_pesq_score
            v_step += 1
            print(f"Epoch: {epoch} | VAL_STEP: {v_step} | VAL_PESQ: {original_pesq(val_pesq_score)}")
        pesq /= v_step

        wandb.log({ 
            "epoch":epoch,
            "val_pesq":original_pesq(pesq),
            "train_PESQ":train_ep_PESQ
        })   
        
        print(f"Epoch:{epoch} | VAL_PESQ:{original_pesq(pesq)}")

        return REWARDS, original_pesq(pesq)

    def train(self, args):
        """
        Run epochs, collect validation results and save checkpoints. 
        """
        best_pesq = -1
        print("Start training...")
        for epoch in range(args.epochs):
            ep_reward, epoch_pesq = self.train_one_epoch(epoch+1)
         
            wandb.log({"Epoch":epoch+1,
                       "Epoch_mean_reward":np.mean(ep_reward)})
            
            if epoch_pesq >= best_pesq:
                best_pesq = epoch_pesq
                #TODO:Logic for savecheckpoint
                if self.gpu_id == 0:
                    checkpoint_prefix = f"{args.exp}_PESQ_{epoch_pesq}_epoch_{epoch}.pt"
                    path = os.path.join(args.output, args.exp, checkpoint_prefix)
                    if self.args.method == 'reinforce':
                        save_dict = {'actor_state_dict':self.actor.state_dict(), 
                                    'optim_state_dict':self.optimizer.state_dict()
                                    }
                    if self.args.method == 'PPO':
                        save_dict = {'actor_state_dict':self.actor.state_dict(), 
                                    'critic_state_dict':self.critic.state_dict(),
                                    'optim_state_dict':self.optimizer.state_dict()
                                    }
                    torch.save(save_dict, path)
                

    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int, args):
    if args.parallel:
        ddp_setup(rank, world_size)
        if rank == 0:
            print(args)
            available_gpus = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            print(f"Available gpus:{available_gpus}")

        train_ds, test_ds = load_data(args.root, 
                                    1, 
                                    1, 
                                    args.cut_len,
                                    gpu = True)
    else:
        if args.gpu:
            gpu = True
        else:
            gpu = False
        train_ds, test_ds = load_data(args.root, 
                                    args.batchsize, 
                                    1, 
                                    args.cut_len,
                                    gpu = False)
    
    pretrain=False
    if args.ckpt is not None:
        pretrain=True

    trainer = Trainer(train_ds, test_ds, args, rank, pretrain=pretrain)
    
    trainer.train(args)
    destroy_process_group()


if __name__ == "__main__":
    ARGS = args().parse_args()

    output = f"{ARGS.output}/{ARGS.exp}"
    os.makedirs(output, exist_ok=True)

    world_size = torch.cuda.device_count()
    print(f"World size:{world_size}")
    if ARGS.parallel:
        mp.spawn(main, args=(world_size, ARGS), nprocs=world_size)
    else:
        if ARGS.gpu:
            main(0, world_size, ARGS)
        else:
            main(None, world_size, ARGS)