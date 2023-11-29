# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
Created on 23rd Nov, 2023
"""

import torch
import torch.nn as nn


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

class RewardModel(nn.Module):
    def __init__(self, ndf, in_channel=2, gpu_id=None):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 2),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.batch_norm = nn.BatchNorm1d(1)

    def forward(self, wav_in, wav_out):
        
        dist = self.batch_norm(self.layers(wav_in) - self.layers(wav_out))
        print(f"dist:{dist.mean()}, {dist.shape}")
        logits = self.classification_head(dist)
        probs = self.softmax(logits)

        return probs
    
class LossNet(nn.Module):
    def __init__(self, in_channels, n_layers, kernel_size, keep_prob, norm_type='sbn'):
        super().__init__()
        self.net = nn.ModuleList()
        self.n_layers = n_layers
        t, f = 401, 201
        for i in range(n_layers):
            out_channels = 32 * (2 ** (i // 5))
            prev_out = 32 * (2 ** ((i-1) // 5))
            if t%2 == 1:
                t = t // 2 + 1
            else:
                t = t // 2
            if f%2 == 1:
                f = f // 2 + 1
            else:
                f = f // 2
            if i == 0:
                if norm_type == 'sbn':
                    layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.BatchNorm2d(out_channels, track_running_stats=False),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob),
                    )
                if norm_type == 'ln':
                    layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.LayerNorm([out_channels, f, t]),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob),
                    )
                if norm_type == 'none':
                    layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.Dropout(1 - keep_prob),
                        nn.LeakyReLU(0.2),
                    )

            elif i == n_layers - 1:
                if norm_type == 'sbn':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, kernel_size, 2, padding=1),
                        nn.BatchNorm2d(out_channels, track_running_stats=False),
                        nn.LeakyReLU(0.2),
                    )
                
                if norm_type == 'ln':
                    layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.LayerNorm([out_channels, f, t]),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob),
                    )
                
                if norm_type == 'none':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, kernel_size, 2, padding=1),
                        nn.LeakyReLU(0.2),
                    )
            else:
                if norm_type == 'sbn':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, kernel_size, 2, padding=1),
                        nn.BatchNorm2d(out_channels, track_running_stats=False),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob),
                    )
                if norm_type == 'ln':
                    layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.LayerNorm([out_channels, f, t]),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob),
                    )
                
                if norm_type == 'none':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, kernel_size, 2, padding=1),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob)
                    )
            self.net.append(layer)
        
    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            outs.append(x)
        return outs

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense3 = nn.Linear(in_dim, 16)
        self.dense4 = nn.Linear(16, 6)
        self.dense2 = nn.Linear(6, out_dim)
        self.relu = nn.LeakyReLU(0.2)
        if out_dim == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim = -1)
        self.outputs = out_dim

    def forward(self, x):
        out = self.relu(self.dense3(x))
        out = self.relu(self.dense4(out))
        out = self.dense2(out)
        if self.outputs == 1:
            scores = self.sigmoid(out)
        if self.outputs > 1:
            scores = self.softmax(out)
        return scores

                
class FeatureLossBatch(nn.Module):
    def __init__(self, n_layers, base_channels, sum_till=14, weights=False, gpu_id=None):
        super().__init__()
        self.out_channels = [base_channels * (2 ** (i // 5)) for i in range(n_layers)]
        self.sum_last_layers=sum_till
        self.n_layers = n_layers
        if weights:
            self.weights = [nn.Parameter(torch.randn(features), requires_grad=True) for features in self.out_channels]
            if gpu_id is not None:
                self.weights = [param.to(gpu_id) for param in self.weights]
        else:
            self.weights = None

    def forward(self, embeds1, embeds2):
        """
        Both embeds1 and embeeds are outputs from each layer of
        loss_net. 
        """
        loss_final = 0
        for i, (e1, e2) in enumerate(zip(embeds1, embeds2)):
            if i >= self.n_layers - self.sum_last_layers:
                dist = e1 - e2
                dist = dist.permute(0, 3, 2, 1)
                if self.weights is not None:
                    res = (self.weights[i] * dist)
                else:
                    res = dist
                loss = torch.mean(res, dim=[1, 2, 3])
                loss_final += loss
        return loss_final


class JNDModel(nn.Module):
    def __init__(self, in_channels, out_dim=2, n_layers=14, keep_prob=0.7, norm_type='sbn', sum_till=14, gpu_id=None):
        super().__init__()
        self.loss_net_real = LossNet(in_channels=in_channels // 2, 
                                     n_layers=n_layers, 
                                     kernel_size=3, 
                                     keep_prob=keep_prob, 
                                     norm_type=norm_type)
        
        self.loss_net_imag = LossNet(in_channels=in_channels // 2, 
                                     n_layers=n_layers, 
                                     kernel_size=3, 
                                     keep_prob=keep_prob, 
                                     norm_type=norm_type)

        self.classification_layer = ClassificationHead(in_dim=2, out_dim=out_dim)

        self.feature_loss = FeatureLossBatch(n_layers=n_layers,
                                             base_channels=32,
                                             gpu_id=gpu_id,
                                             weights=True,
                                             sum_till=sum_till)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ref, inp):

        ref_real = ref[:, 0, :, :].unsqueeze(1)
        ref_imag = ref[:, 1, :, :].unsqueeze(1)

        inp_real = inp[:, 0, :, :].unsqueeze(1)
        inp_imag = inp[:, 1, :, :].unsqueeze(1)

        ref_real = self.loss_net_real(ref_real)
        inp_real = self.loss_net_real(inp_real)

        ref_imag = self.loss_net_imag(ref_imag)
        inp_imag = self.loss_net_imag(inp_imag)

        dist_real = self.feature_loss(ref_real, inp_real).unsqueeze(-1)
        dist_imag = self.feature_loss(ref_imag, inp_imag).unsqueeze(-1)
        
        dist = torch.cat([dist_real, dist_imag], dim=1)
        logits = self.classification_layer(dist.reshape(-1, 2))
        
        return logits
    
