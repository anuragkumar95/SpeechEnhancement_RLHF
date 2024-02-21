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

    def run_episode(self, batch, model):
        """
        Runs an epoch using REINFORCE.
        """
        #Preprocessed batch
        cl_aud, _, noisy, _ = batch

        #Forward pass through model to get the action(mask)
        noisy = noisy.permute(0, 1, 3, 2)
        action, log_probs = model.get_action(noisy)

        #Forward pass through expert model
        exp_action, _ = self.expert.get_action(noisy)

        if self.dist == False:
            #Add gaussian noise
            m_action, log_prob = self.gaussian_noise.get_action_from_raw_action(action[0], t=self.t)
            action = (m_action, action[-1])
            self.t += 1

        else:
            if self.train_phase:
                #finetune both mag and phase
                log_prob = log_prob[0] + log_prob[1][:, 0, :, :].unsqueeze(1) + log_prob[1][:, 1, :, :].unsqueeze(1)
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

        return loss.mean(), reward.mean(), G.mean()

'''
class A3C:
    def __init__(self, gpu_id, beta=0.01, init_model=None, discount=1.0, **params):
        self.env = SpeechEnhancementAgent(n_fft=params['env_params'].get("n_fft"),
                                          hop=params['env_params'].get("hop"),
                                          gpu_id=gpu_id,
                                          args=params['env_params'].get("args"))
        self.discount = discount
        self.gpu_id = gpu_id
        self.expert = init_model.to(self.gpu_id)
        #self.kl_div = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.beta = beta

    def run_episode(self, batch, actor, critic):
        """
        Runs an epoch using REINFORCE.
        """
       #Preprocess batch
        cl_aud, _, noisy = batch

        #Forward pass through expert to get the action(mask)
        noisy = noisy.permute(0, 1, 3, 2)
        action, log_probs, _ = model.get_action(noisy)

        #Forward pass through expert model
        exp_action, _, _ = self.expert.get_action(noisy)

        #Apply mask to get the next state
        next_state = self.env.get_next_state(state=noisy, action=action)
        next_state['cl_audio'] = cl_aud

        #Apply exp_mask to get next state
        exp_next_state = self.env.get_next_state(state=noisy, action=exp_action)
        next_state['exp_est_audio'] = exp_next_state['est_audio']
    
'''
