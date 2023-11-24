# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
Created on 23rd Nov, 2023
"""
from models.reward_model import  RewardModel

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



def ARGS():
    pass

def main(args):
    pass

if __name__=='__main__':
    args = ARGS().parse_args()
    main(args)