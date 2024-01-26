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
import SpeechEnhancementAgent from speech_enh_env

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class ALGORITHMS:
    def __init__(self, **kwargs):
        self.env = SpeechEnhancementAgent(window=kwargs.get("win_len") // 2, 
                                          buffer_size=kwargs.get("b_size"),
                                          n_fft=kwargs.get("n_fft"),
                                          hop=kwargs.get("hop"),
                                          gpu_id=kwargs.get("gpu_id"),
                                          args=kwargs.get("args"))
        pass

    def REINFORCE(self, dataset, model):
        