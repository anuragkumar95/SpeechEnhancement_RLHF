# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.actor import TSCNet
from model.critic import QNet
import os
from data.dataset import load_data
import torch.nn.functional as F
import torch
from utils import power_compress, power_uncompress, batch_pesq, copy_weights, freeze_layers, original_pesq
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
                        help="Path to saved cmgan checkpoint for resuming training.")
    parser.add_argument("--disc_pt", type=str, required=False, default=None,
                        help="Path to saved cmgan checkpoint for discriminator.")
    parser.add_argument("--epochs", type=int, required=False, default=5,
                        help="No. of epochs to be trained.")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--t_max", type=int, required=False, default=4,
                        help="Backpropagate every t_max steps.")
    parser.add_argument("--val_step", type=int, required=False, default=100,
                        help="Run validation every val_step steps.")
    parser.add_argument("--gpu", action='store_true',
                        help="Set this flag for single gpu training.")
    parser.add_argument("--parallel", action='store_true',
                        help="Set this flag for parallel gpu training.")
    parser.add_argument("--reward", type=int, help="Type of reward")
    parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
    parser.add_argument("--win_len", type=int, default=24, help="Context window length for input")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--gamma", type=float, default=0.99, help="Reward discount factor")
    parser.add_argument("--tau", type=float, default=0.99, help="target critic soft update factor")
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                    "and dereverberation")
    return parser

wandb.login()


class DDPGTrainer:
    """
    Class which defines Deep Deterministic Policy Gradient (DDPG)
    Performs Actor Critic training using DDPG method.
    """
    def __init__(self, train_ds, test_ds, args, gpu_id, pretrain=False):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.actor = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1, 
                            win_len=args.win_len,
                            gpu_id=gpu_id)
        self.target_actor = TSCNet(num_channel=64, 
                                   num_features=self.n_fft // 2 + 1, 
                                   win_len=args.win_len,
                                   gpu_id=gpu_id)
        
        if pretrain:
            #Load checkpoint
            print(f"Loading checkpoint saved at {args.ckpt}...")
            cmgan_state_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
            #Copy weights and freeze weights which are copied
            keys, self.actor = copy_weights(cmgan_state_dict, self.actor)
            self.actor = freeze_layers(self.actor, keys)
            keys, self.actor = copy_weights(cmgan_state_dict, self.target_actor)
            self.target_actor = freeze_layers(self.target_actor, keys)
            #Free mem
            del cmgan_state_dict

        self.critic = QNet(ndf=16, in_channel=2, gpu_id=gpu_id)
        self.target_critic = QNet(ndf=16, in_channel=2, gpu_id=gpu_id)

        if args.disc_pt is not None:
            state_dict = torch.load(args.disc_pt, map_location=torch.device('cpu'))
            self.critic.load_state_dict(state_dict['discriminator_state_dict'])
            self.critic = freeze_layers(self.critic, 'all')            

        self.a_optimizer = torch.optim.AdamW(filter(lambda layer:layer.requires_grad,self.actor.parameters()), lr=args.init_lr)
        self.c_optimizer = torch.optim.AdamW(filter(lambda layer:layer.requires_grad,self.critic.parameters()), lr=2 * args.init_lr)

        if gpu_id is not None:
            self.actor = self.actor.to(gpu_id)
            self.critic = self.critic.to(gpu_id)
            self.target_actor = self.target_actor.to(gpu_id)
            self.target_critic = self.target_critic.to(gpu_id)
            
            if args.parallel:
                self.actor = DDP(self.actor, device_ids=[gpu_id])#, find_unused_parameters=True)
                self.critic = DDP(self.critic, device_ids=[gpu_id])
                self.target_actor = DDP(self.target_actor, device_ids=[gpu_id])#, find_unused_parameters=True)
                self.target_critic = DDP(self.target_critic, device_ids=[gpu_id])
            
        self.gpu_id = gpu_id
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
            clean_real : (b * 1 * f * t) real part of clean spectrogram
            clean_imag : (b * 1 * f * t) imag part of clean spectrogram
            clean_mag  : (b * 1 * f * t) mag of clean spectrogram
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
        est_real = noisy_spec[:, 0, :, :].unsqueeze(1)
        est_imag = noisy_spec[:, 1, :, :].unsqueeze(1)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)

        return noisy_spec, clean_spec, clean_real, clean_imag, clean_mag, est_real, est_imag, est_mag, clean, noisy
    
    def preprocess_batch(self, batch):
        """
        Converts a batch of audio waveforms and returns a batch of
        spectrograms.
        ARGS:
            batch : (b * cut_len) waveforms.

        Returns:
            Dict of spectrograms
        """
        clean, noisy, _ = batch
        if self.gpu_id is not None:
            clean = clean.to(self.gpu_id)
            noisy = noisy.to(self.gpu_id)

        noisy_spec, clean_spec, clean_real, clean_imag, clean_mag, est_real, est_imag, est_mag, cl_aud, noisy = self.get_specs(clean, noisy)
        
        ret_val = {'noisy':noisy_spec,
                   'clean':clean_spec,
                   'clean_real':clean_real,
                   'clean_imag':clean_imag,
                   'clean_mag':clean_mag,
                   'cl_audio':cl_aud,
                   'n_audio':noisy,
                   'est_audio':noisy,
                   'est_real':est_real.permute(0, 1, 3, 2),
                   'est_imag':est_imag.permute(0, 1, 3, 2),
                   'est_mag':est_mag.permute(0, 1, 3, 2)
                  }
        
        return ret_val
    
    def train_one_episode(self, epoch, env, args):
        """
        Runs an episode which takes input a batch and predicts masks
        sequentially over the time dimension
        
        actor : actor model that learns P(a|s, theta), where theta are the 
                actor's weight parameters. In our use case it takes windows 
                of spectrogram and predicts masks both in real and complex domain.
        critic: critic model learns to predict the quality of a (s_t, a_t) pair. 

        ARGS:
            batch   : batch of spectrograms of shape (b * 2 * f * t)
            rewards : global ist to store cummulative reward
            args    : global args 
        """
        rewards = []
        torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
      
        for step in range(env.steps-1):
            #get the window input
            inp = env.get_state_input(env.state, step)
            #Forward pass through actor to get the action(mask)
            action = self.actor(inp)
            #Add noise to the action
            action = env.noise.get_action(action)

            #Apply mask to get the next state
            next_state = env.get_next_state(state=env.state, 
                                            action=action, 
                                            t=step)
            
            if next_state is None:
                continue

            #Calculate the reward
            reward = env.get_reward(env.state, next_state)
            rewards.append(reward.mean().detach().cpu().numpy())
            
            #Store the experience in replay_buffer 
            env.exp_buffer.push(state={k:v.detach().clone().cpu().numpy() for k, v in env.state.items()}, 
                                action=(action[0].detach().clone().cpu().numpy(), action[1].detach().clone().cpu().numpy()), 
                                reward=reward.detach().clone().cpu().numpy(), 
                                next_state={k:v.detach().clone().cpu().numpy() for k, v in next_state.items()},
                                t=step)
            
            if len(env.exp_buffer) < args.batchsize:
                continue

            #sample experience from buffer
            experience = env.exp_buffer.sample(args.batchsize)

            #--------------------------- Update Critic ------------------------#
            next_t = experience['t'] + 1
            next_inp = env.get_state_input(experience['next'], next_t)
            next_action = self.target_actor(next_inp)
            next_action = (next_action[0].detach(), next_action[1].detach())
            #Set TD target
            value_next = self.target_critic(experience['next'], next_action, next_t).detach()
            y_t = experience['reward'] + args.gamma * value_next
            value_curr = self.critic(experience['curr'], experience['action'], experience['t'])
            #critic loss
            critic_loss = F.mse_loss(y_t, value_curr).mean()
            self.c_optimizer.zero_grad()
            critic_loss.backward()
            self.c_optimizer.step()
            
            #--------------------------- Update Actor ------------------------#
            #actor loss
            a_inp = env.get_state_input(experience['curr'], experience['t'])
            a_action = self.actor(a_inp)
            actor_loss = -self.critic(experience['curr'], a_action, experience['t']).sum() #+ mag_loss.mean()
            
            self.a_optimizer.zero_grad()
            actor_loss.backward()
            self.a_optimizer.step()

            #update target networks
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * args.tau + target_param.data * (1.0 - args.tau))
        
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * args.tau + target_param.data * (1.0 - args.tau))
        
            #update state
            if reward > 0:
                env.state = next_state
            
            clean = next_state['cl_audio'].cpu().numpy()
            est = next_state['est_audio'].cpu().numpy()
            p_mask, p_score = batch_pesq(clean, est)
            train_pesq = (p_mask * p_score)

            wandb.log({
                'episode_step':step,
                'train_pesq':original_pesq(train_pesq).mean(),
                'actor_loss':actor_loss,
                'critic_loss':critic_loss,
                'y_t':y_t.mean(),
                'current':value_curr.mean(),
                'reward':reward.mean()
            })
            print(f"EPOCH:{epoch} | STEP:{step} | PESQ:{original_pesq(train_pesq).mean()} | REWARD:{reward.mean()}")

        return rewards, actor_loss, critic_loss
    
    def run_validation(self, env):
        """
        Runs a vlidation loop for a batch.
        Predict mask for each frame one at a time 
        and return pesq score of the enhances batch of 
        spectrograms.
        """
        print("Running validation...")
        for step in range(env.steps):
            #get the window input
            inp = env.get_state_input(env.state, step)
            #Forward pas through actor to get the action(mask)
            action = self.actor(inp)
            #Apply action  to get the next state
            next_state = env.get_next_state(state=env.state, 
                                            action=action, 
                                            t=step)
            env.state = next_state

        pesq, pesq_mask = batch_pesq(env.state['cl_audio'].detach().cpu().numpy(), 
                          env.state['est_audio'].detach().cpu().numpy())
        return (pesq*pesq_mask).mean()
    
    def train_one_epoch(self, epoch, args):
        """
        Wrapper function to run one epoch of DDPG.
        One epoch is defined as running one episode of training
        for all batches in the dataset.
        """
        REWARD_MAP=[]
        actor_epoch_loss = 0
        critic_epoch_loss = 0
        step = 0
        env = SpeechEnhancementAgent(window=args.win_len // 2, 
                                     buffer_size=1000000,
                                     n_fft=self.n_fft,
                                     hop=self.hop,
                                     gpu_id=self.gpu_id,
                                     args=args)
        VAL_STEP = args.val_step
        for i, batch in enumerate(self.train_ds):
            self.actor.train()
            self.critic.train()
            self.target_actor.eval()
            self.target_critic.eval()
            #Preprocess batch
            batch = self.preprocess_batch(batch)
            #Run episode
            env.set_batch(batch)
            ep_rewards, actor_loss, critic_loss = self.train_one_episode(epoch, env, args)
            
            #Collect reward and losses
            actor_epoch_loss += actor_epoch_loss
            critic_epoch_loss += critic_epoch_loss
            REWARD_MAP.append(np.mean(ep_rewards))
            step = i+1

            if step % VAL_STEP == 0:
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

            wandb.log({"episode":step,
                       "mean_episode_reward":np.mean(ep_rewards)})
            print(f"Epoch:{epoch} Step:{step+1}: ActorLoss:{actor_loss} CriticLoss:{critic_loss}")
        print(f"Epoch:{epoch}, ActorLoss:{actor_epoch_loss/step}, CriticLoss:{critic_epoch_loss/step}")
        return REWARD_MAP, actor_epoch_loss, critic_epoch_loss, pesq
    
    def train(self, args):
        """
        Run epochs, collect validation results and save checkpoints. 
        """
        best_pesq = -1
        print("Start training...")
        for epoch in range(args.epochs):
            re_map, epoch_actor_loss, epoch_critic_loss,epoch_pesq = self.train_one_epoch(epoch, args)
            #TODO:Log these in wandb
            wandb.log({"Epoch":epoch,
                       "Actor_loss":epoch_actor_loss,
                       "Critic_loss":epoch_critic_loss,
                       "PESQ":epoch_pesq,
                       "reward":np.mean(re_map)})
            
            if epoch_pesq >= best_pesq:
                best_pesq = epoch_pesq
                #TODO:Logic for savecheckpoint
                if self.gpu_id == 0:
                    checkpoint_prefix = f"{args.exp}_PESQ_{epoch_pesq}_epoch_{epoch}.pt"
                    path = os.path.join(args.output, checkpoint_prefix)
                    save_dict = {'actor_state_dict':self.actor.state_dict(), 
                                'critic_state_dict':self.critic.state_dict(),
                                'target_actor_state_dict':self.target_actor.state_dict(),
                                'target_critic_state_dict':self.target_critic.state_dict(),
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
                                    1, 
                                    1, 
                                    args.cut_len,
                                    gpu = False)
    
    pretrain=False
    if args.ckpt is not None:
        pretrain=True

    trainer = DDPGTrainer(train_ds, test_ds, args, rank, pretrain=pretrain)
    
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