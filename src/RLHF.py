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

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class REINFORCE:
    def __init__(self, gpu_id, discount=1.0, **params):
        self.env = SpeechEnhancementAgent(n_fft=params['env_params'].get("n_fft"),
                                          hop=params['env_params'].get("hop"),
                                          gpu_id=gpu_id,
                                          args=params['env_params'].get("args"))
        self.discount = discount
        self.gpu_id = gpu_id
        
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
        #self.env.set_batch(batch)
        cl_aud, clean, noisy = batch
        #Forward pass through expert to get the action(mask)
        noisy = noisy.permute(0, 1, 3, 2)
        print(f"Inp:{noisy.shape}")
        action, log_probs = model.get_action(noisy)

        #Apply mask to get the next state
        next_state = self.env.get_next_state(state=noisy, action=action)
        next_state['cl_audio'] = cl_aud
        
        #Get the reward
        reward = self.env.get_reward(next_state, next_state)
        reward = reward.reshape(-1, 1)
        print(f"Reward:{reward}")
        g_t = self.get_expected_reward(reward)
        print(f"G_t:{g_t}")

        #whitening rewards
        G = torch.tensor(g_t).to(self.gpu_id)
        G = (G - G.mean())/G.std()

        m_lprob, c_lprob = log_probs
        c_lprob = c_lprob.permute(0, 1, 3, 2)
        
        log_prob = m_lprob + c_lprob[:, 0, :, :] + c_lprob[:, 1, :, :] 
        
        print(f"G:{G.shape}, log_probs:{log_prob.shape}")
        loss = (-G * log_prob).sum(-1)
        print(f"LOSS:{loss.shape}")
        return loss, reward
    

