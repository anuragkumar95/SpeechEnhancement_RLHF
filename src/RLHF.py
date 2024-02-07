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
from speech_enh_env import  SpeechEnhancementAgent
import torch


import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class REINFORCE:
    def __init__(self, gpu_id, beta=0.01, init_model=None, discount=1.0, **params):
        self.env = SpeechEnhancementAgent(n_fft=params['env_params'].get("n_fft"),
                                          hop=params['env_params'].get("hop"),
                                          gpu_id=gpu_id,
                                          args=params['env_params'].get("args"))
        self.discount = discount
        self.gpu_id = gpu_id
        self.expert = init_model.to(self.gpu_id)
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')
        self.beta = beta

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
        #Preprocess batch
        cl_aud, _, noisy = batch

        #Forward pass through expert to get the action(mask)
        noisy = noisy.permute(0, 1, 3, 2)
        action, log_probs, _ = model.get_action(noisy)

        #Forward pass through expert model
        _, exp_log_probs, _ = self.expert.get_action(noisy)

        #Apply mask to get the next state
        next_state = self.env.get_next_state(state=noisy, action=action)
        next_state['cl_audio'] = cl_aud

        #Get the reward
        reward = self.env.get_reward(next_state, next_state)

        #Calculate KL_penalty
        if self.expert is not None:
            m_logprob, _ = log_probs
            exp_m_logprob, _ = exp_log_probs
            kl_div_penalty = self.kl_div(m_logprob, exp_m_logprob, log_target=True)
        
        G = reward - self.beta * kl_div_penalty
        G = G.reshape(-1, 1)

        #Ignore complex action, just tune magnitude mask
        m_lprob, _ = log_probs

        loss = []
        for i in range(G.shape[0]):
            loss.append(G[i, ...] * m_lprob[i, ...])
        loss = torch.stack(loss)

    
        return loss.mean(), reward.sum()
    

