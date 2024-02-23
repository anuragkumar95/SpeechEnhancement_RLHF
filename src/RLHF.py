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
from utils import preprocess_batch, power_compress, power_uncompress, batch_pesq, copy_weights, freeze_layers, original_pesq
import logging
from torchinfo import summary
import argparse
import wandb
import psutil
import numpy as np
import traceback
from speech_enh_env import  SpeechEnhancementAgent, GaussianStrategy
import torch


import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class REINFORCE:
    def __init__(self, init_model, reward_model, gpu_id=None, beta=0.01, discount=1.0, **params):
        self.env = SpeechEnhancementAgent(n_fft=params['env_params'].get("n_fft"),
                                          hop=params['env_params'].get("hop"),
                                          gpu_id=gpu_id,
                                          args=params['env_params'].get("args"),
                                          reward_model=reward_model)
        self.discount = discount
        self.gpu_id = gpu_id
        self.expert = init_model.to(self.gpu_id)
        self.rlhf = True
        if reward_model is None:
            self.rlhf = False
        #self.kl_div = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.beta = beta
        self.gaussian_noise = GaussianStrategy(gpu_id=gpu_id)
        self.t = 0
        self.dist = params['env_params'].get("args").out_dist
        self.train_phase = params['train_phase']

    def get_expected_reward(self, rewards):
        """
        Expects rewards to be a numpy array.
        """
        G_t = torch.zeros(rewards.shape).to(self.gpu_id)
        episode_len = rewards.shape[1]
        for i in range(episode_len):
            #Base case: G(T) = r(T)
            #Recursive: G(t) = r(t) + G(t+1)*DISCOUNT
            r_t = rewards[:, episode_len - i - 1]
            if i == 0:
                G_t[:, episode_len - i - 1] = r_t
            else:
                G_t[:, episode_len - i - 1] = r_t + G_t[:, episode_len - i] * self.discount
        return G_t

    def run_episode(self, batch, model, optimizer):
        """
        Runs an epoch using REINFORCE.
        """
        #Preprocessed batch
        cl_aud, _, noisy, _ = batch

        #Forward pass through model to get the action(mask)
        noisy = noisy.permute(0, 1, 3, 2)
        action, log_probs, _ = model.get_action(noisy)

        #Forward pass through expert model
        exp_action, _, _ = self.expert.get_action(noisy)

        if self.dist == False:
            #Add gaussian noise
            m_action, log_prob = self.gaussian_noise.get_action_from_raw_action(action[0], t=self.t)
            action = (m_action, action[-1])
            self.t += 1

        else:
            if self.train_phase:
                #finetune both mag and phase
                log_prob = log_probs[0] + log_probs[1][:, 0, :, :].permute(0, 2, 1) + log_probs[1][:, 1, :, :].permute(0, 2, 1)
            else:  
                #ignore complex mask, just tune mag mask 
                log_prob = log_probs[0]
            

        #Apply mask to get the next state
        a_t = (action[0], exp_action[-1])
        next_state = self.env.get_next_state(state=noisy, action=a_t)
        next_state['cl_audio'] = cl_aud

        #Apply exp_mask to get next state
        exp_next_state = self.env.get_next_state(state=noisy, action=exp_action)
        next_state['exp_est_audio'] = exp_next_state['est_audio']

        #Get the reward
        if not self.rlhf:
            reward, baseline = self.env.get_reward(next_state, next_state)
            G = reward - baseline
        else:
            reward = self.env.get_reward(next_state)
            G = reward
        
        G = G.reshape(-1, 1)
        print(f"G:{G.mean().item()}")

        loss = []
        alpha = 1
        for i in range(G.shape[0]):
            loss.append(alpha * -G[i, ...] * log_prob[i, ...] )
        loss = torch.stack(loss)

        print(f"M_LPROB:{log_prob.mean()}")
        print(f"LOSS:{loss.mean().item()}")

        #Update network
        if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

        return loss.mean(), reward.mean(), G.mean()

class PPO:
    """
    Base class for PPO ploicy gradient method
    """
    def __init__(self, init_model, reward_model, gpu_id=None, beta=0.2, val_coef=0.02, en_coef=0.01, discount=1.0, **params):
        self.env = SpeechEnhancementAgent(n_fft=params['env_params'].get("n_fft"),
                                          hop=params['env_params'].get("hop"),
                                          gpu_id=gpu_id,
                                          args=params['env_params'].get("args"),
                                          reward_model=reward_model)
        
        self.discount = discount
        self.beta = beta
        self.gpu_id = gpu_id
        self.rlhf = True
        if reward_model is None:
            self.rlhf = False
        self.dist = params['env_params'].get("args").out_dist
        self.train_phase = params['train_phase']
        self.t = 0
        self.init_model = init_model
        self.prev_log_probs = None
        self.val_coef = val_coef
        self.en_coef = en_coef

    def run_episode(self, batch, actor, critic, optimizer):
        """
        Imagine the episode N --> C --> Terminal
        So for target values, we consider Noisy --> Clean --> Terminal
        and for current iteration values we consider Noisy --> Enhanced --> Terminal
        """
        #Preprocessed batch
        cl_aud, clean, noisy, _ = batch
        noisy = noisy.permute(0, 1, 3, 2)
        clean = clean.permute(0, 1, 3, 2)

        #Calculate target values and advantages
        with torch.no_grad():
            #Calculate target values for clean state
            state = {}
            state['est_audio'] = clean
            state['exp_est_audio'] = cl_aud
            state['cl_audio'] = cl_aud
            r_c, _ = self.env.get_PESQ_reward(state)
            tgt_val_C = r_c.reshape(-1, 1)
            value_C = critic(clean).reshape(-1, 1).detach()
            adv_c = tgt_val_C - value_C

            #Calculate target values for noisy state
            state = {}
            state['est_audio'] = noisy
            state['exp_est_audio'] = cl_aud
            state['cl_audio'] = cl_aud
            r_n, _ = self.env.get_PESQ_reward(state)
            tgt_val_N = r_n.reshape(-1, 1) + self.discount * tgt_val_C
            value_N = critic(noisy).reshape(-1, 1).detach()
            adv_n = tgt_val_N - value_N

            target_values = torch.stack([tgt_val_N, tgt_val_C], dim=-1).squeeze(1)
            advantages = torch.stack([adv_n, adv_c], dim=-1).squeeze(1)

        
        #Forward pass through model to get the action(mask)
        action, log_probs, entropy = actor.get_action(noisy)
        values = critic(noisy)
        exp_action, exp_log_probs, _ = self.init_model.get_action(noisy)
        
        #Get next state and reward for the state
        a_t = (action[0], exp_action[-1])
        next_state = self.env.get_next_state(state=noisy, action=a_t)
        next_state['cl_audio'] = cl_aud

        #Get expert output
        exp_next_state = self.env.get_next_state(state=noisy, action=exp_action)
        next_state['exp_est_audio'] = exp_next_state['est_audio']

        if not self.rlhf:
            G, _ = self.env.get_PESQ_reward(next_state)
            #G = reward - baseline
        else:
            G = self.env.get_RLHF_reward(next_state)
            
        #Get previous model log_probs 
        if self.t == 0:
            self.prev_log_probs = (exp_log_probs[0].detach(), exp_log_probs[1].detach())
        
        #ignore complex mask, just tune mag mask 
        log_prob, old_log_prob = log_probs[0], self.prev_log_probs[0]
        logratio = log_prob - old_log_prob 
        ratio = torch.exp(logratio)

        print(f"Ratio:{ratio.shape}")
        print(f"Advantages:{advantages.shape}")
        print(f"Tgt_vals:{target_values.shape}")
        
        #Policy loss
        pg_loss1 = -advantages[:, 0] * ratio
        pg_loss2 = -advantages[:, 0] * torch.clamp(ratio, 1 - self.beta, 1 + self.beta)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        #value_loss
        v_loss = 0.5 * ((target_values[:, 0] - values) ** 2).mean()

        #Entropy loss
        entropy_loss = entropy.mean()

        clip_loss = pg_loss - (self.en_coef * entropy_loss) + (self.val_coef * v_loss) 

        #Update network
        if not (torch.isnan(clip_loss).any() or torch.isinf(clip_loss).any()):
            optimizer.zero_grad()
            clip_loss.backward()
            optimizer.step()

        self.prev_log_probs = (log_probs[0].detach(), log_probs[1].detach())
        self.t += 1
                     
        return clip_loss, G.mean(), G.mean()