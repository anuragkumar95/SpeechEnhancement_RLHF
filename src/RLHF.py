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
import SpeechEnhancementAgent from speech_enh_env

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
        

    def run_episode(self, batch, model):
        """
        Runs an epoch using REINFORCE.
        """
        #Preprocess batch
        self.env.set_batch(batch)

        #Forward pass through expert to get the action(mask)
        action, log_probs = model.get_action(self.env.state['noisy'])

        #Apply mask to get the next state
        next_state = self.env.get_next_state(state=self.env.state, 
                                                action=action)
        
        #Get the reward
        reward = self.env.get_reward(self.env.state, next_state)
        
        loss = (-reward * torch.sum(log_probs, dim=-1)).sum()
        return loss, reward
    

