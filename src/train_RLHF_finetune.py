# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.actor import TSCNet
from model.critic import QNet
from model.reward_model import RewardModel
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
import copy

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
    parser.add_argument("--train_phase", action='store_true',
                        help="Phase is also finetuned using RL.")
    parser.add_argument("--suffix", type=str, required=False, default='',
                        help="Save path suffix")
    parser.add_argument("--method", type=str, default='reinforce', required=False,
                        help="RL Algo to run. Choose between (reinforce/PPO)")
    parser.add_argument("--episode_steps", type=int, default=1, required=False,
                        help="No. of steps in episode to run for PPO")
    
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
    def __init__(self, 
                 train_ds, 
                 test_ds, 
                 args, 
                 gpu_id):
        
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.ACCUM_GRAD = args.accum_grad

        self.actor = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1,
                            distribution="Normal", 
                            gpu_id=gpu_id)
        
        self.expert = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1,
                            distribution="Normal", 
                            gpu_id=gpu_id)
        
        cmgan_expert_checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        self.actor.load_state_dict(cmgan_expert_checkpoint['generator_state_dict']) 
        self.expert.load_state_dict(cmgan_expert_checkpoint['generator_state_dict'])
        
        if args.reward_pt is not None:
            self.reward_model = RewardModel(policy=copy.deepcopy(self.actor))
            reward_checkpoint = torch.load(args.reward_pt, map_location=torch.device('cpu'))
            self.reward_model.load_state_dict(reward_checkpoint)
            self.reward_model = freeze_layers(self.reward_model, 'all')
            self.reward_model.eval()
        else:
            self.reward_model = None
        
        #Freeze complex decoder and reward model
        if not args.train_phase:
            self.actor = freeze_layers(self.actor, ['dense_encoder', 'TSCB_1', 'complex_decoder'])
        
        #Set expert to eval and freeze all layers.
        self.expert = freeze_layers(self.expert, 'all')
        self.expert.eval()
     
        print(f"Loaded checkpoint stored at {args.ckpt}. Resuming training...") 
        del cmgan_expert_checkpoint 
        del reward_checkpoint

        if gpu_id is not None:
            self.actor = self.actor.to(gpu_id)
            self.expert = self.expert.to(gpu_id)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(gpu_id)

        if args.method == 'reinforce':
            
            self.optimizer = torch.optim.AdamW(
                filter(lambda layer:layer.requires_grad,self.actor.parameters()), lr=args.init_lr
            )

            self.trainer = REINFORCE(gpu_id=gpu_id, 
                                    beta = 0.25 , 
                                    init_model=self.expert,
                                    discount=1.0,
                                    episode_len=args.episode_steps,
                                    train_phase=args.train_phase,
                                    reward_model=self.reward_model,
                                    env_params={'n_fft':400,
                                                'hop':100, 
                                                'args':args})
            
        if args.method == 'PPO':
            self.critic = QNet(ndf=16, in_channel=2, out_channel=1)
            self.critic = self.critic.to(gpu_id)
            params = list(self.actor.parameters()) + list(self.critic.parameters())
            self.optimizer = torch.optim.AdamW(
                filter(lambda layer:layer.requires_grad, params), lr=args.init_lr
            )
            self.c_optimizer = torch.optim.AdamW(
                filter(lambda layer:layer.requires_grad, self.critic.parameters()), lr=args.init_lr * 1e02
            )

            self.trainer = PPO(init_model=self.expert, 
                               reward_model=self.reward_model, 
                               gpu_id=gpu_id, 
                               beta=0.3,
                               eps=0.02,
                               val_coef=1.0,
                               en_coef=0.0,
                               discount=0.9,
                               warm_up_steps=0,
                               run_steps=args.episode_steps,
                               train_phase=args.train_phase,
                               accum_grad=args.accum_grad,
                               env_params={'n_fft':400,
                                            'hop':100, 
                                            'args':args})

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

        if self.args.train_phase:
            a_t = action
        else:
            a_t = (action[0], exp_action[-1])
        
        #Apply action  to get the next state
        next_state = env.get_next_state(state=inp, 
                                        action=a_t)

        pesq, pesq_mask = batch_pesq(clean_aud.detach().cpu().numpy(), 
                                     next_state['est_audio'].detach().cpu().numpy())
        return (pesq*pesq_mask).sum()
    

    def train_one_epoch(self, epoch):
        
        #Run training
        self.actor.train()
        if self.args.method == 'PPO':
            self.critic.train()
        REWARDS = []
        num_batches = len(self.train_ds)
        
        for i, batch in enumerate(self.train_ds):   
           
            #Each minibatch is an episode
            batch = preprocess_batch(batch, gpu_id=self.gpu_id) 
            try: 
                if self.args.method == 'reinforce': 
                    loss, batch_reward = self.trainer.run_episode(batch, self.actor, self.optimizer)

                    wandb.log({
                        "episode": (i+1) + ((epoch - 1) * num_batches),
                        "G_t":batch_reward[0].item(),
                        "r_t":batch_reward[1].item(),
                        "cumulative_G_t": batch_reward[0].item() + self.G, 
                        "loss":loss,
                    })

                if self.args.method == 'PPO':
                    loss, batch_reward = self.trainer.run_episode(batch, self.actor, self.critic, (self.optimizer, self.c_optimizer))
                    
                    if loss is not None:
                        wandb.log({
                            "episode": (i+1) + ((epoch - 1) * num_batches),
                            "episode_avg_kl":batch_reward[2].item(),
                            "cumulative_G_t": batch_reward[0].item(),
                            "critic_values": batch_reward[1].item(), 
                            "episodic_avg_r": batch_reward[3].item(),
                            "clip_loss":loss[0],
                            "value_loss":loss[1],
                            "entropy_loss":loss[2]
                        })

            except Exception as e:
                print(traceback.format_exc())
                continue
            
            if loss is not None:
                self.G = batch_reward[0].item() + self.G
                print(f"Epoch:{epoch} | Episode:{i+1} | Return: {batch_reward[0].item()} | Values: {batch_reward[1].item()} | KL: {batch_reward[2].item()}")
                REWARDS.append(batch_reward[0].item())

        #Run validation
        self.actor.eval()
        if self.args.method == 'PPO':
            self.critic.eval()
        pesq = 0
        v_step = 0
        with torch.no_grad():
            for i, batch in enumerate(self.test_ds):
                
                #Preprocess batch
                batch = preprocess_batch(batch, gpu_id=self.gpu_id)
                
                #Run validation episode
                try:
                    val_pesq_score = self.run_validation(self.trainer.env, batch)
                except Exception as e:
                    print(traceback.format_exc())
                    continue

                pesq += val_pesq_score/self.args.batchsize
                v_step += 1
                print(f"Epoch: {epoch} | VAL_STEP: {v_step} | VAL_PESQ: {original_pesq(val_pesq_score/self.args.batchsize)}")
        pesq = pesq / v_step 

        wandb.log({ 
            "epoch":epoch-1,
            "val_pesq":original_pesq(pesq),
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
            
            #if epoch_pesq >= best_pesq:
            #    best_pesq = epoch_pesq
                #TODO:Logic for savecheckpoint
            if self.gpu_id == 0:
                checkpoint_prefix = f"{args.exp}_PESQ_{epoch_pesq}_epoch_{epoch}.pt"
                path = os.path.join(args.output, f"{args.exp}_{args.suffix}", checkpoint_prefix)
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

    trainer = Trainer(train_ds, test_ds, args, rank)
    
    trainer.train(args)
    destroy_process_group()


if __name__ == "__main__":
    ARGS = args().parse_args()

    output = f"{ARGS.output}/{ARGS.exp}_{ARGS.suffix}"
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