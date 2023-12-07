# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
import matplotlib.pyplot as plt

class ContrastiveLoss(nn.Module):
    def __init__(self, reduction='mean', eps=0.0001):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, distances, labels):

        loss = (labels) * (distances ** 2) \
             + (1 - labels) * (torch.clamp(self.eps - distances, min=0.0) ** 2 )
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()


def copy_weights(src_state_dict, target):
    """
    Copy weights from src model to target model.
    Only common layers are transferred.
    ARGS:
        src_state_dict : source model state dict to copy weights from.
        target         : model to copy weights to.

    Returns:
        A list of layers that were copied.
    """
    src_layers = src_state_dict
    target_layers = target.state_dict()
    copied_keys = []
    for src_key, target_key in zip(src_layers, target_layers):
        #If key is empty, it's a description of the entire model, skip this key
        if len(src_key) == 0:
            continue
        #Found a matching key, copy the weights
        elif src_key == target_key : 
            target_layers[target_key].data.copy_(src_layers[src_key].data)
            copied_keys.append(target_key)
    
    #update the state dict of the target model
    target.load_state_dict(target_layers)
    
    return copied_keys, target
        

def freeze_layers(model, layers):
    """
    Freezes specific layers of the model.
    ARGS:
        model : instance of the model.
        layer : list of name of the layers to be froze.
    
    Returns:
        Model instance with frozen parameters.
    """
    for name, param in model.named_parameters():
        if layers == 'all':
            if param.requires_grad:
                param.requires_grad = False
        else:
            for layer in layers:
                if ((layer == name) or (layer in name)) and param.requires_grad:
                    param.requires_grad = False
    return model 