# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import torch
import numpy as np
from utils import batch_pesq, power_uncompress
from collections import deque
import torch.nn.functional as F


class SpeechEnhancementAgent:
    def __init__(self, batch, window, buffer_size, n_fft, hop, args, gpu_id=None):
        """
        State : Dict{noisy, clean, est_real, est_imag, cl_audio, est_audio}
        """
        self.state = batch
        self.clean = batch['clean']
        self.steps = batch['noisy'].shape[2]
        
        self.gpu_id = gpu_id
        self.window = window
        self.n_fft = n_fft
        self.hop = hop
        self.args = args
        
        self.exp_buffer = replay_buffer(buffer_size, gpu_id=gpu_id)
        self.noise = OUNoise(action_dim=batch['noisy'].shape[-1], gpu_id=gpu_id)

    def get_state_input(self, state, t):
        """
        Get the batched windowed input for time index t
        ARGS:
            t : time index

        Returns
            Batch of windowed input centered around t
            of shape (b, 2, f, w) 
        """
        state = state['noisy']
        b, _, tm, f = state.shape
        left = t - self.window
        right = t + self.window + 1
        if t < self.window: 
            pad = torch.zeros(b, 2, -left, f)
            if self.gpu_id is not None:
                pad = pad.to(self.gpu_id)
            windows = torch.cat([pad, state[:, :, :right, :]], dim=2)
        elif right > tm - 1:
            pad = torch.zeros(b, 2, right - tm, f)
            if self.gpu_id is not None:
                pad = pad.to(self.gpu_id)
            windows = torch.cat([state[:, :, left:, :], pad], dim=2) 
        else:
            windows = state[:, :, left:right, :]
        return windows

    def get_next_state(self, state, action, t):
        """
        Apply mask to spectrogram on the i-th frame and return next state.
        ARGS:
            state : spectrograms of shape (b x 2 x f x t)
            action: (mask, complex_mask) for frame at index 't' for entire batch, (b x f x 1)

        Returns:
            Next state with 't'th frame enhanced by applying mask.
        """
        #print(f"State:{state['noisy'].shape}")
        b, _, tm, f = state['noisy'].shape
        mask = torch.ones(b, 1, tm, f)
        complex_mask = torch.ones(b, 2, tm, f)

        if self.gpu_id is not None:
            mask = mask.to(self.gpu_id)
            complex_mask = complex_mask.to(self.gpu_id)
        
        #action = self.noise.get_action(action)
        mask_mag, complex_out = action
        
        #Output mask is for the 't'th frame of the window
        #print(mask[:, :, t, :].shape, mask_mag.squeeze(2).shape, complex_mask[:, :, :, t].shape, complex_out.squeeze(-1).shape)
        mask[:, :, t, :] = mask_mag.squeeze(2)
        complex_mask[:, :, t, :] = complex_out.squeeze(-1)

        mag = (state['noisy'][:, 0, :, :] ** 2) + (state['noisy'][:, 1, :, :] ** 2)
        mag = torch.sqrt(mag)
        mag = mag.unsqueeze(1)
        
        noisy_phase = torch.angle(
            torch.complex(state['noisy'][:, 0, :, :], state['noisy'][:, 1, :, :])
        ).unsqueeze(1)

        #print(mask.shape,mag.shape)
        out_mag = mask * mag
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        est_real = mag_real + complex_mask[:, 0, :, :].unsqueeze(1)
        est_imag = mag_imag + complex_mask[:, 1, :, :].unsqueeze(1)
        est_real = est_real.permute(0, 1, 3, 2)
        est_imag = est_imag.permute(0, 1, 3, 2)

        window = torch.hamming_window(self.n_fft)
        if self.gpu_id is not None:
            window = window.to(self.gpu_id)

        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        #print(est_spec_uncompress.shape)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=window,
            onesided=True,
        )

        next_state = torch.cat([est_real, est_imag], dim=1).permute(0, 1, 3, 2)
        clean_mag = torch.sqrt(state['clean_real']**2 + state['clean_imag']**2)

        retval = {'noisy':next_state.detach(),
                  'clean':state['clean'].detach(), 
                  'clean_real':state['clean_real'].detach(),
                  'clean_imag':state['clean_imag'].detach(),
                  'clean_mag':clean_mag.detach(),
                  'cl_audio':state['cl_audio'].detach(),
                  'n_audio':state['n_audio'].detach(),
                  'est_mag':est_mag.detach(),
                  'est_real':est_real.detach(),
                  'est_imag':est_imag.detach(),
                  'est_audio':est_audio.detach()
                  }
        return retval

    def get_reward(self, state, next_state):
        """
        Calculate the reward of the current state.
        Reward is defined as the tanh of relative difference between 
        PESQ scores of the noisy and the current enhanced signal.

        R(t) = tanh(z'-z), this is bounded to be in the range(-1, 1).
        """
        if self.args.reward == 1:
            z_mask, z = batch_pesq(state['cl_audio'].detach().cpu().numpy(), 
                                state['est_audio'].detach().cpu().numpy())
            z_hat_mask, z_hat = batch_pesq(next_state['cl_audio'].detach().cpu().numpy(), 
                                        next_state['est_audio'].detach().cpu().numpy())
            pesq_reward = (z_hat_mask * z_hat) - (z_mask * z)

            if self.gpu_id is not None:
                pesq_reward = pesq_reward.to(self.gpu_id)
            return pesq_reward
        
        if self.args.reward == 2:
            z_mask, z = batch_pesq(next_state['cl_audio'].detach().cpu().numpy(),
                                   next_state['n_audio'].detach().cpu().numpy())
            
            z_hat_mask, z_hat = batch_pesq(next_state['cl_audio'].detach().cpu().numpy(), 
                                           next_state['est_audio'].detach().cpu().numpy())
            
            pesq_reward = (z_hat_mask * z_hat) - (z_mask * z)

            if self.gpu_id is not None:
                pesq_reward = pesq_reward.to(self.gpu_id)
            return pesq_reward
        
        if self.args.reward == 3:
            z_mask, z = batch_pesq(next_state['cl_audio'].detach().cpu().numpy(),
                                   next_state['n_audio'].detach().cpu().numpy())
            
            z_hat_mask, z_hat = batch_pesq(next_state['cl_audio'].detach().cpu().numpy(), 
                                           next_state['est_audio'].detach().cpu().numpy())
            
            pesq_reward = (z_hat_mask * z_hat) - (z_mask * z)

            if self.gpu_id is not None:
                pesq_reward = pesq_reward.to(self.gpu_id)

            loss_mag = F.mse_loss(next_state['clean_mag'], next_state['est_mag']).detach()
            loss_real = F.mse_loss(next_state['clean_real'],next_state['est_real']).detach()
            time_loss = F.mse_loss(next_state['cl_audio'], next_state['est_audio']).detach()

            r_t = pesq_reward - torch.tanh(self.args.loss_weights[0]*loss_real + 
                                            self.args.loss_weights[1]*loss_mag + 
                                            self.args.loss_weights[2]*time_loss) 
            return r_t     
    

class replay_buffer:
    def __init__(self, max_size, gpu_id=None):
        self.buffer = deque(maxlen=max_size)
        self.gpu_id = gpu_id

    def push(self, state, action, reward, next_state, t):
        experience = {'curr':state,
                      'action':action,
                      'reward':reward,
                      'next':next_state, 
                      't':t}
        self.buffer.append(experience)

    def sample(self):
        idx = np.random.choice(len(self.buffer), 1)[0]
        if self.gpu_id is None:
            retval = {'curr':{k:torch.FloatTensor(v) for k, v in self.buffer[idx]['curr'].items()},
                      'next':{k:torch.FloatTensor(v) for k, v in self.buffer[idx]['next'].items()},
                      'action':(torch.FloatTensor(self.buffer[idx]['action'][0]),
                                torch.FloatTensor(self.buffer[idx]['action'][1])),
                      'reward':torch.FloatTensor(self.buffer[idx]['reward']),
                      't':self.buffer[idx]['t']
                     }
        else:
            retval = {'curr':{k:torch.FloatTensor(v).to(self.gpu_id) for k, v in self.buffer[idx]['curr'].items()},
                      'next':{k:torch.FloatTensor(v).to(self.gpu_id) for k, v in self.buffer[idx]['next'].items()},
                      'action':(torch.FloatTensor(self.buffer[idx]['action'][0]).to(self.gpu_id),
                                torch.FloatTensor(self.buffer[idx]['action'][1]).to(self.gpu_id)),
                      'reward':torch.FloatTensor(self.buffer[idx]['reward']).to(self.gpu_id),
                      't':self.buffer[idx]['t']
                     }
        return retval

    def __len__(self):
        return len(self.buffer)

# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000, gpu_id=None):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.gpu_id = gpu_id
        self.reset()
        
        
    def reset(self):
        self.state = torch.ones(self.action_dim) * self.mu
        if self.gpu_id is not None:
            self.state = self.state.to(self.gpu_id)
        
    def evolve_state(self, action):
        x  = self.state
        action_dim = action[0].shape[-1]
        rand = torch.randn(action_dim)
        if self.gpu_id is not None:
            rand = rand.to(self.gpu_id)
        dx = self.theta * (self.mu - x) + self.sigma * rand
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state(action)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        mag_mask = torch.clip(action[0] + ou_state, torch.tensor(0.0).to(self.gpu_id), torch.max(action[0]))
        comp_mask = torch.clip(action[1] + ou_state.view(-1, 1), torch.min(action[1]), torch.max(action[1]))
        return (mag_mask, comp_mask)