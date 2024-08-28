# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import shutil

import numpy as np
from pesq import pesq
#import matplotlib.pyplot as plt

"""
class KL_Divergence(nn.Module):
    def __init__(self, reduction='batchmean'):
        self.reduction = reduction

    def forward(self, input_mu, input_var, target_mu, target_var):
        batch, _, _, _ = input_mu.shape

        inp_mu = input_mu.reshape(batch, -1)
        inp_sigma = (input_var ** 2).reshape(batch, -1)
        trgt_mu = target_mu.reshape(batch, -1)
        trgt_sigma = (target_var ** 2).reshape(batch, -1)

        #covariance matrices
        cov_1 = torch.diag_embed(inp_sigma)
        cov_2 = torch.diag_embed(trgt_sigma)

        #calculate trace
        product = torch.bmm(torch.linalg.inv(cov_1), cov_2)
"""







def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = []
    for c,n in zip(clean, noisy):
        pesq = pesq_loss(c, n)
        pesq_score.append(pesq)
    #Mask invalid pesq scores
    score_mask = np.array([1 if pqs > -1 else 0 for pqs in pesq_score])
    pesq_score = np.array(pesq_score)
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(score_mask), torch.FloatTensor(pesq_score)

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


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    

class K_way_CrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        """
        Calculates K-way ordinal cross-entropy loss
        ARGS:
            logits : [-1, K] logits.
            target : [-1, K, K] shaped one-hot vector.

        Returns:
            K-way ordinal cross-entropy loss
        """
        classes = logits.shape[-1]
        log_preds = torch.log(logits)
        
        log_likelihoods = 0
        for k in range(classes):
            k_targets = target[:, k, :]
            log_likelihood = k_targets * log_preds
            log_likelihoods += log_likelihood.sum(-1)

        if self.reduction == 'mean':        
            return -log_likelihoods.mean()
        else:
            return -log_likelihoods.sum()

def get_specs_1(wav, n_fft, hop, gpu_id=None):
    """
    Create spectrograms from input waveform.
    ARGS:
        wav : waveform (batch * cut_len)
    """
    win = torch.hamming_window(n_fft)

    if gpu_id is not None:
        win = win.to(gpu_id)
    
    spec = torch.stft(
        wav,
        n_fft,
        hop,
        window=win,
        onesided=True,
    )
    return spec 

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

def original_pesq(pesq):
    return (pesq * 3.5) + 1



def get_spec_and_phase(signal, n_fft, hop, gpu_id):
    stft = torch.stft(
        signal,
        n_fft,
        hop,
        n_fft,
        torch.hamming_window(512).to(gpu_id),
        center=True,
        pad_mode='constant',
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    stft = stft.transpose(2, 1)

    phase = torch.atan2(stft[:, :, :, 1], stft[:, :, :, 0])
    feat = spectral_magnitude(stft, power=0.5)
    feat = torch.log1p(feat)
    return feat, phase

def spectral_magnitude(stft, power=1, log=False, eps=1e-14):
    spectr = stft.pow(2).sum(-1)

    # Add eps avoids NaN when spectr is zero
    if power < 1:
        spectr = spectr + eps
    spectr = spectr.pow(power)

    if log:
        return torch.log(spectr + eps)
    return spectr

def transform_spec_to_wav(mag, phase, signal_length=None):
    # Combine with enhanced magnitude
    complex_predictions = torch.mul(
        torch.unsqueeze(mag, -1),
        torch.cat(
            (
                torch.unsqueeze(torch.cos(phase), -1),
                torch.unsqueeze(torch.sin(phase), -1),
            ),
            -1,
        ),
    )
    complex_predictions = complex_predictions.permute(0, 2, 1, 3)
    complex_predictions = torch.complex(complex_predictions[..., 0], complex_predictions[..., 1])
    
    pred_wavs = torch.istft(
        input=complex_predictions,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hamming_window(512).to(complex_predictions.device),
        center=True,
        onesided=True,
        normalized=False,
        length=signal_length,
    )
    return pred_wavs



def get_specs(clean, noisy, gpu_id, n_fft, hop, compress=True, ref=None, clean_istft=False, return_c=False):
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

    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    if clean is not None:
        clean = torch.transpose(clean, 0, 1)
        clean = torch.transpose(clean * c, 0, 1)
    
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
    if clean is not None:
        clean_spec = torch.stft(
            clean,
            n_fft,
            hop,
            window=win,
            onesided=True,
        )
    else:
        clean_spec = None

    if compress:
        noisy_spec = power_compress(noisy_spec)#.permute(0, 1, 3, 2)

        if clean_spec is not None:
            clean_spec = power_compress(clean_spec)
    
    if clean_istft:
        #Take istft for clean_spec to account for changes in stft to istft
        clean_spec_uncompress = power_uncompress(clean_spec[:, 0, :, :], clean_spec[:, 1, :, :]).squeeze(1)
        
        clean_audio = torch.istft(
            clean_spec_uncompress,
            n_fft,
            hop,
            window=win,
            onesided=True,
        )

        clean = torch.transpose(clean_audio, 0, 1)
        clean = torch.transpose(clean * c, 0, 1)
    
    if clean is None:
        clean = noisy

    if ref is not None:
        ref = torch.transpose(ref, 0, 1)
        ref = torch.transpose(ref * c, 0, 1)

        ref_spec = torch.stft(
            ref,
            n_fft,
            hop,
            window=win,
            onesided=True,
        )
        ref_spec = power_compress(ref_spec)
        
        return clean, clean_spec, noisy_spec, ref_spec
    
    if return_c:
        return clean, clean_spec, noisy_spec, c
    
    return clean, clean_spec, noisy_spec

def preprocess_batch(batch, ref=None, n_fft=400, hop=100, gpu_id=None, clean_istft=False, return_c=False, model='cmgan'):
    """
    Converts a batch of audio waveforms and returns a batch of
    spectrograms.
    ARGS:
        batch : (b * cut_len) waveforms.

    Returns:
        List of spectrograms
    """
    clean, noisy, labels = batch

    if gpu_id is not None:
        if clean is not None:
            clean = clean.to(gpu_id)
        noisy = noisy.to(gpu_id)

    if model == 'metricgan':
        noisy_spec, _ = get_spec_and_phase(noisy, n_fft=n_fft, hop=hop, gpu_id=gpu_id) 
        clean_spec, _ = get_spec_and_phase(clean, n_fft=n_fft, hop=hop, gpu_id=gpu_id)
        print(f"INP:{noisy_spec.shape}, {clean_spec.shape}")
        return (clean, clean_spec.unsqueeze(1), noisy_spec.unsqueeze(1), labels, None)       


    if ref is not None:
        ref = ref.to(gpu_id)
        clean, clean_spec, noisy_spec, ref_spec = get_specs(clean, noisy, gpu_id, n_fft=n_fft, hop=hop, ref=ref, clean_istft=clean_istft)
        return (clean, clean_spec, noisy_spec, ref_spec, labels)
    
    if return_c:
        clean, clean_spec, noisy_spec, c = get_specs(clean, noisy, gpu_id, n_fft=n_fft, hop=hop, ref=ref, clean_istft=clean_istft, return_c=return_c)
        return (clean, clean_spec, noisy_spec, labels, c)
    
    clean, clean_spec, noisy_spec = get_specs(clean, noisy, gpu_id, n_fft=n_fft, hop=hop, clean_istft=clean_istft)
    return (clean, clean_spec, noisy_spec, labels)

def map_state_dict(checkpoint):

    map_ = {"mask_decoder.mask_conv.4.weight":"mask_decoder.final_conv.weight",
            "mask_decoder.mask_conv.4.bias":"mask_decoder.final_conv.bias"}

    for key in map_:
        checkpoint[map_[key]] = checkpoint[key]
        checkpoint.pop(key)

    return checkpoint

