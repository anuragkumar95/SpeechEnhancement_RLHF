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
    def __init__(self, gpu_id, discount=1.0, **env_params):
        self.env = SpeechEnhancementAgent(n_fft=env_params.get("n_fft"),
                                          hop=env_params.get("hop"),
                                          gpu_id=gpu_id,
                                          args=env_params.get("args"))
        self.discount = discount
        self.gpu_id = gpu_id
        
    def get_expected_reward(self, rewards):
        """
        Expects rewards to be a numpy array.
        """
        G_t = np.zeros(rewards.shape)
        episode_len = rewards.shape[1]
        for i in range(episode_len):
            #Base case: G(T) = r(T)
            #Recursive: G(t) = r(t) + G(t+1)*DISCOUNT
            r_t = rewards[:, episode_len - i - 1]
            if i == 0:
                G_t[:, episode_len - i - 1] = r_t
            else:
                G_t[:, episode_len - i - 1] = r_t + G_t[:, episode_len - i] * self.discount
        return np.array(G_t)

    def run_episode(self, batch, model):
        """
        Runs an epoch using REINFORCE.
        """
        #Preprocess batch
        #self.env.set_batch(batch)

        #Forward pass through expert to get the action(mask)
        action, log_probs = model.get_action(self.env.state['noisy'])

        #Apply mask to get the next state
        next_state = self.env.get_next_state(state=self.env.state, 
                                                action=action)
        
        #Get the reward
        reward = self.env.get_reward(self.env.state, next_state)
        reward = reward.reshape(-1, 1)
        g_t = self.get_expected_reward(reward)

        #whitening rewards
        G = torch.tensor(g_t).to(self.gpu_id)
        G = (G - G.mean())/G.std()
        
        loss = (-G * torch.sum(log_probs, dim=-1)).sum()
        return loss, reward
    

