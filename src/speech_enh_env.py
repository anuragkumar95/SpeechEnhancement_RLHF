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
    def __init__(self, batch, window, buffer_size, gpu_id=None):
        """
        State : Dict{noisy, clean, est_real, est_imag, cl_audio, est_audio}
        """
        self.state = batch
        self.clean = batch['clean']
        self.window = window
        self.steps = batch['noisy'].shape[2]
        self.gpu_id = gpu_id
        self.exp_buffer = replay_buffer(buffer_size)
        self.noise = OUNoise(action_dim=batch['noisy'].shape[-1])

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
        print(f"State:{state.shape}")
        b, _, tm, f = state.shape
        left = t - self.window
        right = t + self.window + 1
        if t < self.window // 2 : 
            pad = torch.zeros(b, 2, -left, f)
            if self.gpu_id is not None:
                pad = pad.to(self.gpu_id)
            windows = torch.cat([pad, state[:, :, left:right, :]], dim=2)
        elif right > tm - 1:
            pad = torch.zeros(b, 2, right - tm, f)
            if self.gpu_id is not None:
                pad = pad.to(self.gpu_id)
            windows = torch.cat([state[:, :, left:right, :], pad], dim=2) 
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
        b, _, f, t = state['noisy'].shape
        mask = torch.ones(b, 1, f, t)
        complex_mask = torch.ones(b, 1, f, t)

        if self.gpu_id is not None:
            mask = mask.to(self.gpu_id)
            complex_mask = complex_mask.to(self.gpu_id)
        
        #action = self.noise.get_action(action)
        mask_mag, complex_out = action
        
        #Output mask is for the 't'th frame of the window
        mask[:, :, :, t] = mask_mag
        complex_mask[:, :, :, t] = complex_out

        mag = torch.sqrt(state['noisy'][:, 0, :, :] ** 2 + state['noisy'][:, 1, :, :] ** 2).unsqueeze(1)
        
        noisy_phase = torch.angle(
            torch.complex(state[:, 0, :, :], state[:, 1, :, :])
        ).unsqueeze(1)

        out_mag = mask * mag
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        est_real = mag_real + complex_mask[:, 0, :, :].unsqueeze(1)
        est_imag = mag_imag + complex_mask[:, 1, :, :].unsqueeze(1)

        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft),#.to(self.gpu_id),
            onesided=True,
        )

        next_state = torch.cat([est_real, est_imag], dim=1)
        clean_mag = torch.sqrt(state['clean_real']**2 + state['clean_imag']**2)

        retval = {'noisy':next_state,
                  'clean_real':state['clean_real'],
                  'clean_imag':state['clean_imag'],
                  'clean_mag':clean_mag,
                  'cl_audio':state['cl_audio'],
                  'est_mag':est_mag,
                  'est_real':est_real,
                  'est_imag':est_imag,
                  'est_audio':est_audio
                  }
        return retval

    def get_reward(self, state, next_state):
        """
        Calculate the reward of the current state.
        Reward is defined as the tanh of relative difference between 
        PESQ scores of the noisy and the current enhanced signal.

        R(t) = tanh(z'-z), this is bounded to be in the range(-1, 1).
        """
        z = batch_pesq(state['clean'], state['noisy'])
        z_hat = batch_pesq(next_state['clean'], next_state['noisy'])

        pesq_reward = z_hat - z

        loss_mag = F.mse_loss(next_state['clean_mag'], next_state['est_mag'])   
        loss_real = F.mse_loss(next_state['clean_real'],next_state['est_real'])
        loss_imag = F.mse_loss(next_state['clean_imag'], next_state['est_imag'])
        time_loss = F.mse_loss(next_state['cl_audio'], next_state['est_audio'])

        r_t = torch.tanh(pesq_reward - (loss_mag + loss_real + loss_imag + time_loss)) 
        return r_t    
    

class replay_buffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, t):
        experience = {'curr':state,
                      'action':action,
                      'reward':reward,
                      'next':next_state, 
                      't':t}
        self.buffer.append(experience)

    def sample(self):
        idx = np.random.choice(len(self.buffer), 1)[0]
        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer)

# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self, action):
        x  = self.state
        action_dim = action[0].shape[-1]
        print(f"X:{x.shape}, act_dim:{action_dim}")
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state(action)
        print(f"ou_state:{ou_state.shape}")
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        mag_mask = np.clip(action[0] + ou_state, torch.min(action[0]), torch.max(action[0]))
        comp_mask = np.clip(action[1] + ou_state, torch.min(action[1]), torch.max(action[1]))
        return (mag_mask, comp_mask)