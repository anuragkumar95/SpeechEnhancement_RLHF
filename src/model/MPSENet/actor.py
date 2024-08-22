
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.MPSENet.conformer import ConformerBlock

from torch.distributions import Normal

#from utils import get_padding_2d, LearnableSigmoid_2d
from pesq import pesq
from joblib import Parallel, delayed


#Taken from https://github.com/yxlu-0102/MP-SENet/blob/main/models/generator.py

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))


class LearnableSigmoid_1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class DenseBlock(nn.Module):
    def __init__(self, dense_channel=64, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        #self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(dense_channel*(i+1), dense_channel, kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(dense_channel, affine=True),
                nn.PReLU(dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, dense_channel=64):
        super(DenseEncoder, self).__init__()
        #self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

        self.dense_block = DenseBlock(dense_channel, depth=4) # [b, h.dense_channel, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, n_fft, beta, dense_channel=64, out_channel=1, gpu_id=None, eval=False):
        super(MaskDecoder, self).__init__()
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            #nn.Conv2d(out_channel, out_channel, (1, 1))
        )
      
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.lsigmoid = LearnableSigmoid_2d(n_fft//2+1, beta=beta)
        self.gpu_id = gpu_id
        self.evaluation = eval

    def sample(self, mu, logvar, x=None):
        #if self.dist == 'Normal':
        #    sigma = torch.clamp(torch.exp(logvar) + 1e-08, min=1.0)
        #elif self.dist is None:
        sigma = (torch.ones(mu.shape)*0.01).to(self.gpu_id) 
        N = Normal(mu, sigma)
        if x is None:
            x = N.rsample()
        if len(x.shape) != len(mu.shape):
            raise ValueError(f"No. of dims in action {x.shape} don't match mu {mu.shape}")
        if x.shape[-1] != mu.shape[-1]:
            x = x.permute(0, 1, 3, 2)
        x_logprob = N.log_prob(x)
        x_entropy = N.entropy()
        return x, x_logprob, x_entropy, (mu, sigma)

    def forward(self, x, action=None):
        
        x = self.dense_block(x)
        x = self.mask_conv(x)
        
        x_mu = self.final_conv(x)
        x, x_logprob, x_entropy, params = self.sample(x_mu, None, action)
        if self.evaluation:
            x_out = self.lsigmoid(params[0].permute(0, 3, 2, 1).squeeze(-1))
            x_out = x_out.permute(0, 2, 1).unsqueeze(1)
        else:
            x_out = x.permute(0, 3, 2, 1).squeeze(-1)
            x_out = self.lsigmoid(x_out).permute(0, 2, 1).unsqueeze(1)
            #x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)

        return (x, x_out), x_logprob, x_entropy, params


class PhaseDecoder(nn.Module):
    def __init__(self, dense_channel, out_channel=1, gpu_id=None, eval=False):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 1))
     
        self.evaluation = eval
        self.gpu_id = gpu_id

    def sample(self, mu, logvar, x=None):
       
        sigma = (torch.ones(mu.shape)*0.01).to(self.gpu_id) 
        N = Normal(mu, sigma)
        if x is None:
            x = N.rsample()
        if len(x.shape) != len(mu.shape):
            raise ValueError(f"No. of dims in action {x.shape} don't match mu {mu.shape}")
        if x.shape[-1] != mu.shape[-1]:
            x = x.permute(0, 1, 3, 2)
        x_logprob = N.log_prob(x)
        x_entropy = N.entropy()
        return x, x_logprob, x_entropy, (mu, sigma)

    def forward(self, x, action=None):
        
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r_mu = self.phase_conv_r(x)
        x_i_mu = self.phase_conv_i(x)
        x_r, x_r_logprob, x_r_entropy, r_params = self.sample(x_r_mu, None, action)
        x_i, x_i_logprob, x_i_entropy, i_params = self.sample(x_i_mu, None, action)
        if self.evaluation:
            x_r = r_params[0]
            x_i = i_params[0]
        x = torch.atan2(x_r, x_i)

        x_logprob = torch.stack([x_r_logprob, x_i_logprob], dim=1).squeeze(2)
        x_entropy = torch.stack([x_r_entropy, x_i_entropy], dim=1).squeeze(2)
        params = (torch.stack([r_params[0], i_params[0]], dim=1), torch.stack([r_params[1], i_params[0]], dim=1))

        return x, x_logprob, x_entropy, params


class TSConformerBlock(nn.Module):
    def __init__(self, dense_channel):
        super(TSConformerBlock, self).__init__()
        #self.h = h
        self.time_conformer = ConformerBlock(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class MPNet(nn.Module):
    def __init__(self, n_fft, beta, dense_channel, num_tscblocks=4, gpu_id=None, eval=False):
        super(MPNet, self).__init__()
        #self.h = h
        self.num_tscblocks = num_tscblocks
        self.dense_encoder = DenseEncoder(dense_channel=dense_channel, in_channel=2)

        self.TSConformer = nn.ModuleList([])
        for i in range(num_tscblocks):
            self.TSConformer.append(TSConformerBlock(dense_channel=dense_channel))
        
        self.mask_decoder = MaskDecoder(n_fft=n_fft, beta=beta, dense_channel=dense_channel, out_channel=1, gpu_id=gpu_id, eval=eval)
        self.phase_decoder = PhaseDecoder(dense_channel=dense_channel, out_channel=1, gpu_id=gpu_id, eval=eval)

    def set_evaluation(self, bool):
        self.mask_decoder.evaluation = bool
        self.phase_decoder.evaluation = bool

    def get_action(self, x):
        #b, ch, t, f = x.size()
        noisy_mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_pha = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
    
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSConformer[i](x)
       
        mask, m_logprob, m_entropy, params = self.mask_decoder(x)
        complex_out, c_logprob, c_entropy, c_params = self.phase_decoder(x)

        m_logprob = m_logprob.squeeze(1)
        c_logprob = c_logprob.permute(0, 1, 3, 2)

        return (mask, complex_out), (m_logprob, c_logprob), (m_entropy, c_entropy), (params, c_params)

    
    def get_action_prob(self, x, action=None):
        """
        ARGS:
            x : spectrogram
            action : (Tuple) Tuple of mag and complex actions

        Returns:
            Tuple of mag and complex masks log probabilities.
        """
        noisy_mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        
        noisy_pha = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)

        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSConformer[i](x)

        if action is not None:
            m_action = action[0][0]
            if len(m_action.shape) == 3:
                m_action = m_action.unsqueeze(1)
            c_action = action[1]
            if len(c_action.shape) == 3:
                c_action = c_action.unsqueeze(1)
      
        _, m_logprob, m_entropy, _ = self.mask_decoder(x, m_action)
        _, c_logprob, c_entropy, _ = self.phase_decoder(x, c_action)

        m_logprob = m_logprob.squeeze(1)
        c_logprob = c_logprob.permute(0, 1, 3, 2)

        #print(f"m_log:{m_logprob.shape}, c_log:{c_logprob.shape}")

        return (m_logprob, c_logprob), (m_entropy, c_entropy)
        

    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSConformer[i](x)

        (_, mask), _, _, _ = self.mask_decoder(x)
        complex_out, _, _, _ = self.phase_decoder(x)

        denoised_mag = (noisy_mag * mask).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = complex_out.permute(0, 3, 2, 1).squeeze(-1)
        denoised_com = torch.stack((denoised_mag*torch.cos(denoised_pha),
                                    denoised_mag*torch.sin(denoised_pha)), dim=-1)

        return denoised_mag, denoised_pha, denoised_com


def phase_losses(phase_r, phase_g, h):

    dim_freq = h.n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))

    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            h.sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1

    return pesq_score