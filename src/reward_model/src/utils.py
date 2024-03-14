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
        if labels.shape[-1] > 1:
            labels = torch.argmax(labels, dim=-1).reshape(-1, 1)
       
        loss = (1 - labels) * (distances ** 2) \
             + (labels) * (torch.clamp(self.eps - distances, min=0.0) ** 2 )
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

def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)

def get_specs(clean, noisy, gpu_id, n_fft, hop):
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
    
    win = torch.hamming_window(n_fft)
    if gpu_id is not None:
        win = win.to(gpu_id)

    noisy_spec = torch.stft(
        noisy,
        n_fft,
        hop,
        window=win,
        onesided=True,
    )
    clean_spec = torch.stft(
        clean,
        n_fft,
        hop,
        window=win,
        onesided=True,
    )

    noisy_spec = power_compress(noisy_spec)#.permute(0, 1, 3, 2)
    clean_spec = power_compress(clean_spec)
    return clean, clean_spec, noisy_spec