# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import torch
import numpy as np
from utils import batch_pesq, power_uncompress
from collections import deque
import torch.nn.functional as F
#import gym
#from gym import Env, spaces

"""
class GymSpeechEnhancementEnv(Env):
    def __init__(self, spec_shape, mask_shape, low, high, win_len, n_fft=400, hop=100):
        super().__init__()
        self.state_shape = spec_shape
        self.observation_space = spaces.Box(low=low, 
                                            high=high,
                                            shape=self.state_shape)
        
        self.action_space = spaces.Box(low=low, 
                                       high=high,
                                       shape=mask_shape)
        
        self.agent = SpeechEnhancementAgent(window=win_len // 2, 
                                            buffer_size=250,
                                            n_fft=n_fft,
                                            hop=hop)

        pass

    def step(self, action):
        state = self.agent.get_next_state(self, state, action, t)

    def reset(self):
        pass
"""
class SpeechEnhancementAgent:
    def __init__(self, window, buffer_size, n_fft, hop, args, gpu_id=None):
        """
        State : Dict{noisy, clean, est_real, est_imag, cl_audio, est_audio}
        """
        self.gpu_id = gpu_id
        self.window = window
        self.n_fft = n_fft
        self.hop = hop
        self.args = args
        self.exp_buffer = replay_buffer(buffer_size, gpu_id=gpu_id)
        

    def set_batch(self, batch):
        self.state = batch
        self.clean = batch['clean']
        self.steps = batch['noisy'].shape[2]
        self.noise = OUNoise(action_dim=batch['noisy'].shape[-1], gpu_id=self.gpu_id)

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
        windows = []
        for i in range(t.shape[0]):
            if t[i] < self.window: 
                pad = torch.zeros(b, 2, -left[i], f)
                if self.gpu_id is not None:
                    pad = pad.to(self.gpu_id)
                win = torch.cat([pad, state[:, :, :right[i], :]], dim=2)
            elif right[i] > tm - 1:
                pad = torch.zeros(b, 2, right[i] - tm, f)
                if self.gpu_id is not None:
                    pad = pad.to(self.gpu_id)
                win = torch.cat([state[:, :, left[i]:, :], pad], dim=2) 
            else:
                win = state[:, :, left[i]:right[i], :]
            windows.append(win)
        windows = torch.stack(windows).squeeze(1)
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
        try:
            b, _, tm, f = state['noisy'].shape
            mask = torch.ones(b, 1, tm, f)
            complex_mask = torch.zeros(b, 2, tm, f)

            if self.gpu_id is not None:
                mask = mask.to(self.gpu_id)
                complex_mask = complex_mask.to(self.gpu_id)
            
            #action = self.noise.get_action(action)
            mask_mag, complex_out = action
            
            #Output mask is for the 't'th frame of the window
            mask[:, :, t, :] = mask_mag.squeeze(2)
            complex_mask[:, :, t, :] = complex_out.squeeze(-1)

            mag = (state['noisy'][:, 0, :, :] ** 2) + (state['noisy'][:, 1, :, :] ** 2)
            mag = torch.sqrt(mag)
            mag = mag.unsqueeze(1)
            
            noisy_phase = torch.angle(
                torch.complex(state['noisy'][:, 0, :, :], state['noisy'][:, 1, :, :])
            ).unsqueeze(1)

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

        except Exception as e:
            return None

    def get_reward(self, state, next_state):
        """
        Calculate the reward of the current state.
        Reward is defined as the tanh of relative difference between 
        PESQ scores of the noisy and the current enhanced signal.

        R(t) = tanh(z'-z), this is bounded to be in the range(-1, 1).
        """
        if self.args.reward == 0:
            z_hat_mask, z_hat = batch_pesq(next_state['cl_audio'].detach().cpu().numpy(), 
                                        next_state['est_audio'].detach().cpu().numpy())
            pesq_reward = (z_hat_mask * z_hat)

            if self.gpu_id is not None:
                pesq_reward = pesq_reward.to(self.gpu_id)
            return pesq_reward.mean()
        
        if self.args.reward == 1:
            z_mask, z = batch_pesq(state['cl_audio'].detach().cpu().numpy(), 
                                state['est_audio'].detach().cpu().numpy())
            z_hat_mask, z_hat = batch_pesq(next_state['cl_audio'].detach().cpu().numpy(), 
                                        next_state['est_audio'].detach().cpu().numpy())
            pesq_reward = (z_hat_mask * z_hat) - (z_mask * z)

            if self.gpu_id is not None:
                pesq_reward = pesq_reward.to(self.gpu_id)
            return pesq_reward.mean()
        
        if self.args.reward == 2:
            z_mask, z = batch_pesq(next_state['cl_audio'].detach().cpu().numpy(),
                                   next_state['n_audio'].detach().cpu().numpy())
            
            z_hat_mask, z_hat = batch_pesq(next_state['cl_audio'].detach().cpu().numpy(), 
                                           next_state['est_audio'].detach().cpu().numpy())
            
            pesq_reward = (z_hat_mask * z_hat) - (z_mask * z)

            if self.gpu_id is not None:
                pesq_reward = pesq_reward.to(self.gpu_id)
            return torch.tanh(pesq_reward).mean()
        
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

            r_t = torch.tanh(pesq_reward - (self.args.loss_weights[0]*loss_real + 
                                            self.args.loss_weights[1]*loss_mag + 
                                            self.args.loss_weights[2]*time_loss)).mean()
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

    def sample(self, batch_size):
        CURR = {}
        NEXT = {}
        ACTION = [[], []]
        REWARD = []
        T = []
        for _ in range(batch_size):
            idx = np.random.choice(len(self.buffer), 1)[0]
            
            for k, v in self.buffer[idx]['curr'].items():
                if k not in CURR:
                    CURR[k] = []
                v = torch.FloatTensor(v)
                CURR[k].append(v)
            
            for k, v in self.buffer[idx]['next'].items():
                if k not in NEXT:
                    NEXT[k] = []
                v = torch.FloatTensor(v)
                NEXT[k].append(v)

            r = torch.FloatTensor(self.buffer[idx]['reward'])
            REWARD.append(r)

            t = self.buffer[idx]['t']
            T.append(t)

            action = (torch.FloatTensor(self.buffer[idx]['action'][0]), torch.FloatTensor(self.buffer[idx]['action'][1]))
            ACTION[0].append(action[0])
            ACTION[1].append(action[1])

        ACTION = (torch.stack(ACTION[0]), torch.stack(ACTION[1]))
        REWARD = torch.stack(REWARD)
        CURR = {k:torch.stack(v).squeeze(1) for k, v in CURR.items()}
        NEXT = {k:torch.stack(v).squeeze(1) for k, v in NEXT.items()}
        T = np.array(T)

        return {'curr':CURR, 
                'next':NEXT, 
                'action':ACTION, 
                'reward':REWARD, 
                't':T}

    def __len__(self):
        return len(self.buffer)

# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.015, max_sigma=0.05, min_sigma=0.05, decay_period=100000, gpu_id=None):
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