# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.actor import TSCNet
from model.critic import QNet
from model.cmgan import TSCNetExpert
from RLHF import REINFORCE

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
    parser.add_argument("-pt", "--ckpt", type=str, required=False, default=None,
                        help="Path to saved checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, required=False, default=5,
                        help="No. of epochs to be trained.")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--gpu", action='store_true',
                        help="Set this flag for single gpu training.")
    parser.add_argument("--parallel", action='store_true',
                        help="Set this flag for parallel gpu training.")
    
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
    def __init__(self, train_ds, test_ds, args, gpu_id, out_distribution=False, pretrain=False):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.actor = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1,
                            distribution=out_distribution, 
                            gpu_id=gpu_id)
        
        
        cmgan_expert_checkpoint = torch.load(args.expert_pt, map_location=torch.device(gpu_id))
        self.expert.load_state_dict(cmgan_expert_checkpoint)       

        self.a_optimizer = torch.optim.AdamW(filter(lambda layer:layer.requires_grad,self.actor.parameters()), lr=args.init_lr)
        #self.c_optimizer = torch.optim.AdamW(filter(lambda layer:layer.requires_grad,self.critic.parameters()), lr=2 * args.init_lr)

        if gpu_id is not None:
            self.actor = self.actor.to(gpu_id)
            self.critic = self.critic.to(gpu_id)
            self.target_actor = self.target_actor.to(gpu_id)
            self.target_critic = self.target_critic.to(gpu_id)

            self.trainer = REINFORCE(gpu_id=gpu_id, 
                                     optimizer=self.a_optimizer, 
                                     alpha=args.init_lr, 
                                     discount=1.0,
                                     env_params={'n_fft' : 400,
                                                 'hop' : 100,
                                                 'args':args})

            if args.ckpt is not None:
                state_dict = torch.load(args.ckpt, map_location=torch.device(gpu_id))
                self.actor.load_state_dict(state_dict['actor_state_dict'])
                #self.critic.load_state_dict(state_dict['critic_state_dict'])
                self.a_optimizer.load_state_dict(state_dict['actor_optim_state_dict'])
                #self.c_optimizer.load_state_dict(state_dict['critic_optim_state_dict'])
                #_, self.target_actor = copy_weights(state_dict['actor_state_dict'], self.target_actor)
                #_, self.target_critic = copy_weights(state_dict['critic_state_dict'], self.target_critic)
                del state_dict
                print(f"Loaded checkpoint stored at {args.ckpt}. Resuming training...")

            if args.parallel:
                self.actor = DDP(self.actor, device_ids=[gpu_id])
                self.critic = DDP(self.critic, device_ids=[gpu_id])
                self.target_actor = DDP(self.target_actor, device_ids=[gpu_id])
                self.target_critic = DDP(self.target_critic, device_ids=[gpu_id])

        
            
        self.gpu_id = gpu_id
        self.expert.eval()
        self.target_actor.eval()
        self.target_critic.eval()
        wandb.init(project=args.exp)

    '''
    def train_one_step(self, epoch, step, env, args):
        """
        Runs one step
        """
        torch.autograd.set_detect_anomaly(True)
        
        clean = env.state['cl_audio'].detach().cpu().numpy()
        est = env.state['est_audio'].detach().cpu().numpy()
        p_mask, p_score = batch_pesq(clean, est)
        noisy_pesq = original_pesq((p_mask * p_score).mean())

        self.actor.train()

        #Forward pass through expert to get the action(mask)
        action = self.expert.get_action(env.state['noisy'])
        action = (action[0].detach(), action[1].detach())
        
        #Add noise to the action
        #action = env.noise.get_action(action)

        #Apply mask to get the next state
        next_state = env.get_next_state(state=env.state, 
                                        action=action)
        
        #Calculate the reward
        reward = env.get_reward(env.state, next_state)

        #Store the experience in replay_buffer 
        env.exp_buffer.push(state={k:v.detach().cpu().numpy() for k, v in env.state.items()}, 
                            action=(action[0].detach().cpu().numpy(), action[1].detach().cpu().numpy()), 
                            reward=reward.detach().cpu().numpy(), 
                            next_state={k:v.detach().cpu().numpy() for k, v in next_state.items()})
        
        env.state = next_state
        
        del(action)
        del(next_state)

        torch.cuda.empty_cache()
        
        #sample experience from buffer
        experience = env.exp_buffer.sample(args.batchsize)

        #--------------------------- Update Critic ------------------------#
    
        next_action = self.target_actor(experience['next']['noisy'])
        next_action = (next_action[0].detach(), next_action[1].detach())
        
        #Set TD target
        value_next = self.target_critic(experience['next'], next_action).detach()
        y_t = experience['reward'] + args.gamma * value_next
        value_curr = self.critic(experience['curr'], experience['action'])
        
        #critic loss
        critic_loss = F.mse_loss(y_t, value_curr).mean()
        self.c_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 5.0)
        self.c_optimizer.step()

        #--------------------------- Update Actor ------------------------#
        
        #actor loss
        a_action = self.actor(experience['curr']['noisy'])
        actor_loss = -self.critic(experience['curr'], a_action).mean()
        
        self.a_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 5.0)
        self.a_optimizer.step()

        #--------------------- Update Target Networks --------------------#
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * args.tau + target_param.data * (1.0 - args.tau))
    
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * args.tau + target_param.data * (1.0 - args.tau))

        #sample action from actor to measure improvement
        self.actor.eval()
        action = self.actor(env.state['noisy'])
        est_state = env.get_next_state(state=env.state, action=action)

        clean = est_state['cl_audio'].detach().cpu().numpy()
        est = est_state['est_audio'].detach().cpu().numpy()
        p_mask, p_score = batch_pesq(clean, est)
        
        train_pesq = original_pesq((p_mask * p_score).mean())

        torch.cuda.empty_cache()

        print(f"EPOCH:{epoch} | STEP:{step} | change in PESQ:{train_pesq - noisy_pesq} | REWARD:{reward.mean()}")

        outputs = {
            'reward':reward.mean(),
            'actor_loss':actor_loss.detach(),
            'critic_loss':critic_loss.detach(),
            'y_t':y_t.detach(),
            'value_curr':value_curr.detach(),
            'value_next':value_next.detach(),
            'train_pesq':train_pesq,
            'noisy_pesq':noisy_pesq
        }

        return outputs
    '''
    
    def run_validation(self, env):
        """
        Runs a vlidation loop for a batch.
        Predict mask for each frame one at a time 
        and return pesq score of the enhances batch of 
        spectrograms.
        """
        #print("Running validation...")
        
        inp = env.state['noisy']
        #Forward pass through actor to get the action(mask)
        action, _ = self.actor(inp)
        #Apply action  to get the next state
        next_state = env.get_next_state(state=env.state, 
                                        action=action)

        pesq, pesq_mask = batch_pesq(env.state['cl_audio'].detach().cpu().numpy(), 
                          next_state['est_audio'].detach().cpu().numpy())
        return (pesq*pesq_mask).mean()
    
    '''
    def train_one_episode(self, epoch, args):
        """
        Wrapper function to run one epoch of DDPG.
        One epoch is defined as running one episode of training
        for all batches in the dataset.
        """
        actor_epoch_loss = 0
        critic_epoch_loss = 0
        ep_reward = 0
        step = 0
        env = SpeechEnhancementAgent(window=args.win_len // 2, 
                                     buffer_size=1200,
                                     n_fft=self.n_fft,
                                     hop=self.hop,
                                     gpu_id=self.gpu_id,
                                     args=args)

       
        for i, batch in enumerate(self.train_ds):
            self.actor.train()
            self.critic.train()

            #Preprocess batch
            batch = self.preprocess_batch(batch)
            #Run episode  
            env.set_batch(batch)
            step = i+1
            
            outputs = self.train_one_step(epoch, step, env, args)
            if outputs is None:
                continue

            #Collect reward and losses
            actor_epoch_loss += outputs['actor_loss']
            critic_epoch_loss += outputs['critic_loss']
            ep_reward += outputs['reward']

       
        actor_epoch_loss = actor_epoch_loss / step
        critic_epoch_loss = critic_epoch_loss / step
        ep_reward = ep_reward / step
        print(f"Epoch:{epoch} | ActorLoss:{actor_epoch_loss} | CriticLoss:{critic_epoch_loss}")

        #Run validation
        self.actor.eval()
        self.critic.eval()
        pesq = 0
        v_step = 0
        for i, batch in enumerate(self.test_ds):
            #Preprocess batch
            batch = self.preprocess_batch(batch)
            env.set_batch(batch)
            #Run validation episode
            val_pesq_score = self.run_validation(env)
            pesq += val_pesq_score
            v_step += 1
        pesq /= v_step
        wandb.log({"val_step":v_step,
                    "val_pesq":original_pesq(pesq)})   
        
        print(f"Epoch:{epoch} | VAL_PESQ:{original_pesq(pesq)}")

        return ep_reward, actor_epoch_loss, critic_epoch_loss, pesq
    '''
    def train_one_epoch(self, epoch):
        #Train
        self.actor.train()
        self.critic.train()
        REWARDS = []
        num_batches = len(self.train_ds)
        for i, batch in enumerate(self.train_ds):   
            
            #Each minibatch is an episode
            batch = preprocess_batch(batch, gpu_id=self.gpu_id)
            batch_loss, batch_reward = self.trainer.run_episode(self.train_ds, self.actor)

            self.a_optimizer.zero_grad()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.critic.parameters(), 5.0)
            self.a_optimizer.step()

            wandb.log({
                "episode_cumulative_reward":batch_reward,
                "episode": (i+1)+(epoch*num_batches)
            })
            print(f"Epoch:{epoch} | Episode:{i+1} | Reward: {batch_reward}")
            REWARDS.append(batch_reward)

            
        #Run validation
        self.actor.eval()
        self.critic.eval()
        pesq = 0
        v_step = 0
        for i, batch in enumerate(self.test_ds):
            #Preprocess batch
            batch = self.preprocess_batch(batch)
            self.trainer.env.set_batch(batch)
            #Run validation episode
            val_pesq_score = self.run_validation(self.trainer.env)
            pesq += val_pesq_score
            v_step += 1
        pesq /= v_step

        wandb.log({ 
            "epoch":epoch,
            "val_step":v_step,
            "val_pesq":original_pesq(pesq)
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
            ep_reward, epoch_actor_loss, epoch_critic_loss, epoch_pesq = self.train_one_epoch(epoch+1, args)
            #TODO:Log these in wandb
            wandb.log({"Epoch":epoch+1,
                       "Actor_loss":epoch_actor_loss,
                       "Critic_loss":epoch_critic_loss,
                       "ValPESQ":original_pesq(epoch_pesq),
                       "Epoch_mean_reward":ep_reward})
            
            if epoch_pesq >= best_pesq:
                best_pesq = epoch_pesq
                #TODO:Logic for savecheckpoint
                if self.gpu_id == 0:
                    checkpoint_prefix = f"{args.exp}_PESQ_{epoch_pesq}_epoch_{epoch}.pt"
                    path = os.path.join(args.output, args.exp, checkpoint_prefix)
                    save_dict = {'actor_state_dict':self.actor.module.state_dict(), 
                                'critic_state_dict':self.critic.module.state_dict(),
                                'actor_optim_state_dict':self.a_optimizer.state_dict(),
                                'critic_optim_state_dict':self.c_optimizer.state_dict(),
                                #'scheduler_state_dict':scheduler.state_dict(),
                                #'lr':scheduler.get_last_lr()
                                }
                    torch.save(save_dict, path)
                #TODO:May need a LR scheduler as well

    
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
        train_ds, test_ds = load_data(args.root, 
                                    16, 
                                    1, 
                                    args.cut_len,
                                    gpu = False)
    
    pretrain=False
    if args.ckpt is not None:
        pretrain=True

    trainer = Trainer(train_ds, test_ds, args, rank, out_distribution=args.out_dist, pretrain=pretrain)
    
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