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


class TSLossNet(nn.Module):
    def __init__(self, in_channels, n_layers, kernel_size, keep_prob, norm_type='sbn'):
        super().__init__()
        self.net_time = nn.ModuleList()
        self.net_freq = nn.ModuleList()
        self.n_layers = n_layers
    
        t, f = 401, 201
        for i in range(n_layers):
            out_channels = 32 * (2 ** (i // 5))
            
            if t%2 == 1:
                t = t // 2 + 1
            else:
                t = t // 2

            if i < n_layers - 1:
                if norm_type == 'sbn':
                    layer = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.BatchNorm1d(out_channels, track_running_stats=False),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1-keep_prob)
                    )  
                if norm_type == 'ln':
                    layer = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.LayerNorm([out_channels, t]),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1-keep_prob)
                    )
            else:
                if norm_type == 'sbn':
                    layer = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.BatchNorm1d(out_channels, track_running_stats=False),
                        nn.Dropout(1-keep_prob)
                    ) 
                if norm_type == 'ln':
                    layer = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.LayerNorm([out_channels, t]),
                        nn.Dropout(1-keep_prob)
                    )
            in_channels = out_channels
            self.net_time.append(layer)

        for i in range(n_layers):
            out_channels = 32 * (2 ** (i // 5))

            if f%2 == 1:
                f = f // 2 + 1
            else:
                f = f // 2

            if i < n_layers - 1:
                if norm_type == 'sbn':
                    layer = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.BatchNorm1d(out_channels, track_running_stats=False),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1-keep_prob)
                    ) 
                if norm_type == 'ln':
                    layer = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.LayerNorm([out_channels, f]),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1-keep_prob)
                    ) 
            else:
                if norm_type == 'sbn':
                    layer = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.BatchNorm1d(out_channels, track_running_stats=False),
                        nn.Dropout(1-keep_prob)
                    ) 
                if norm_type == 'ln':
                    layer = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, 2, padding=1),
                        nn.LayerNorm([out_channels, f]),
                        nn.Dropout(1-keep_prob)
                    ) 
            in_channels = out_channels
            self.net_freq.append(layer)

    def forward(self, x):
        outs = []
        b, c, t, f = x.shape
        #reshape x (b, ch, t, f) --> x_time (b * f, t, ch)
        x_t = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        #reshape x (b, ch, t, f) --> x_time (b * t, f, ch)
        x_f = x.permute(0, 2, 3, 1).contiguous().view(b * t, f, c)
        for i in range(self.n_layers):
            x_t = self.net_time[i](x_t)
            x_f = self.net_freq[i](x_f)
            outs.append((x_t.view(b, f, t, c), x_f.view(b, t, f, c)))
        return outs

    
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
                        nn.BatchNorm2d(out_channels),
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
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2),
                    )
                
                if norm_type == 'ln':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, kernel_size, 2, padding=1),
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
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(1 - keep_prob),
                    )
                if norm_type == 'ln':
                    layer = nn.Sequential(
                        nn.Conv2d(prev_out, out_channels, kernel_size, 2, padding=1),
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



class TSFeatureLosBatch(nn.Module):
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
                dist_t = e1[0] - e2[0]
                dist_f = e1[1] = e2[1]
                if self.weights is not None:
                    res_t = (self.weights[i] * dist_t)
                    res_f = (self.weights[i] * dist_f)
                else:
                    res_t = dist_t
                    res_f = dist_f
                loss = torch.mean(res_t, dim=[1, 2, 3]) + torch.mean(res_f, dim=[1, 2, 3])
                loss_final += loss
        return loss_final

"""
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(query, key, value)
"""

class AttentionFeatureLossBatch(nn.Module):
    def __init__(self, n_layers, base_channels, time_bins=401, freq_bins=201, sum_till=14, gpu_id=None):
        super().__int__()
        out_channels = [base_channels * (2 ** (i // 5)) for i in range(n_layers)]
        bins = []
        for _ in range(n_layers):
            if time_bins % 2 == 1:
                time_bins = (time_bins // 2) + 1
            else:
                time_bins = time_bins // 2
            if freq_bins % 2 == 1:
                freq_bins = (freq_bins // 2) + 1
            else:
                freq_bins = freq_bins // 2
            bins.append((time_bins, freq_bins))
        self.attention_layers = nn.ModuleList()
        for i in range(n_layers):
            ch, (t, f) = out_channels[i], bins[i]
            
            total_dim = t + t + ch
            time_attn = nn.MultiheadAttention(total_dim, 1)

            total_dim = f + f + ch
            freq_attn = nn.MultiheadAttention(total_dim, 1)

            self.attention_layers.append((time_attn, freq_attn))
        
    def forward(self, embeds1, embeds2):
        loss_final = 0
        for i, (e1, e2) in enumerate(zip(embeds1, embeds2)):
            #both e1 and e2 is of shape (b, ch, t, f)
            b, ch, t, f = e1.shape
            #diff is average difference across time and freq axis
            #should be of shape (b, ch)
            diff = torch.mean((e1 - e2), dim=[2, 3])

            #for time attn, reshape both to (b*f, ch, t)
            e1_t = e1.permute(0, 3, 1, 2).contiguous().view(b * f, ch, t)
            e2_t = e2.permute(0, 3, 1, 2).contiguous().view(b * f, ch, t)
            attn_time_outputs, _ = self.attention_layers[i][0](e1_t, e2_t, diff)
            
            #for freq attn, reshape both to (b*t, ch, f)
            e1_f = e1.permute(0, 2, 1, 3).contiguous().view(b * t, ch, f)
            e2_f = e2.permute(0, 2, 1, 3).contiguous().view(b * t, ch, f)
            attn_freq_outputs, _ = self.attention_layers[i][1](e1_f, e2_f, diff)

            #Average attn outputs across ch and t/f dims
            attn_scores = torch.mean(attn_time_outputs, dim=[1, 2]) + torch.mean(attn_freq_outputs, dim=[1, 2])
            loss_final += attn_scores
        return loss_final




                
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
    def __init__(self, in_channels, out_dim=2, n_layers=14, keep_prob=0.7, loss_type='featureloss', norm_type='sbn', sum_till=14, enc_type=1, gpu_id=None):
        super().__init__()
        if enc_type == 1:
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
            
        if enc_type == 2:
            self.loss_net_real = LossNet(in_channels=in_channels, 
                                         n_layers=n_layers, 
                                         kernel_size=3, 
                                         keep_prob=keep_prob, 
                                         norm_type=norm_type)

        if enc_type == 1:
            self.classification_layer = ClassificationHead(in_dim=2, out_dim=out_dim)
        elif enc_type == 2:
            self.classification_layer = ClassificationHead(in_dim=1, out_dim=out_dim)

        if loss_type == 'featureloss':
            self.feature_loss = FeatureLossBatch(n_layers=n_layers,
                                                base_channels=32,
                                                gpu_id=gpu_id,
                                                weights=True,
                                                sum_till=sum_till)
        
        if loss_type == 'attentionloss':
            self.feature_loss = AttentionFeatureLossBatch(n_layers=n_layers,
                                                        base_channels=32,
                                                        gpu_id=gpu_id,
                                                        weights=True,
                                                        sum_till=sum_till)


        self.sigmoid = nn.Sigmoid()
        self.type = enc_type


    def forward(self, ref, inp):
        if self.type == 1:
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

        if self.type == 2:
            inp = self.loss_net_real(inp)
            ref = self.loss_net_real(ref)

            dist = self.feature_loss(ref, inp)
            logits = self.classification_layer(dist.reshape(-1, 1))
        
        return logits
    
