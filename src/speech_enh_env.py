# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""
import os
import torch
import numpy as np
from utils import batch_pesq, power_uncompress
from collections import deque
import subprocess
import tempfile

import soundfile as sf
import torch.nn.functional as F
from torch.distributions import Normal

#import gym
#from gym import Env, spaces

PI = 3.14

class SpeechEnhancementAgent:
    def __init__(self, n_fft, hop, args, reward_model=None, buffer_size=None, gpu_id=None):
        """
        State : Dict{noisy, clean, est_real, est_imag, cl_audio, est_audio}
        """
        self.gpu_id = gpu_id
        self.n_fft = n_fft
        self.hop = hop
        self.args = args
        self.reward_model = None
        if buffer_size is not None:
            self.exp_buffer = replay_buffer(buffer_size, gpu_id=gpu_id)
        if reward_model is not None:
            self.reward_model = reward_model
        

    #def set_batch(self, batch):
    #    self.state = batch
    #    self.clean = batch['clean']
    #    self.steps = batch['noisy'].shape[2]
    #    #self.noise = OUNoise(action_dim=batch['noisy'].shape[-1], gpu_id=self.gpu_id)
    
       
    def get_next_state(self, state, action):
        """
        Apply mask to spectrogram and return next (enhanced) state.
        ARGS:
            state : spectrograms of shape (b x 2 x f x t)
            action: (mask, complex_mask) for spectrogram.

        Returns:
            Next state enhanced by applying mask.
        """
        x = state
        (_, mask), complex_out = action
        
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)

        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        if mag.shape != mask.shape:
            mask = mask.permute(0, 2, 1).unsqueeze(1)
        out_mag = mask * mag
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)

        est_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        est_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        window = torch.hamming_window(self.n_fft)
        if self.gpu_id is not None:
            window = window.to(self.gpu_id)

        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1).permute(0, 2, 1, 3)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=window,
            onesided=True,
        )
        

        est_spec = torch.stack([est_real, est_imag], dim=1).squeeze(2)
        
        #next_state = {k:v.detach() for k, v in state.items()}
        next_state = {}
        next_state['noisy'] = est_spec
        next_state['est_mag'] = est_mag.permute(0, 1, 3, 2)
        next_state['est_real'] = est_real.permute(0, 1, 3, 2)
        next_state['est_imag'] = est_imag.permute(0, 1, 3, 2)
        next_state['est_audio'] = est_audio

        return next_state
    '''
    def get_RLHF_reward(self, inp, out):
        """
        ARGS:
            inp : spectrogram of curr state (b * ch * t * f) 
            out : spectrogram of next state (b * ch * t * f) 

        Returns
            Reward in the range (0, 1) for next state with reference to curr state.
        """
        return self.reward_model.get_reward(out) - self.reward_model.get_reward(inp) 
    '''
    def get_RLHF_reward(self, state, scale=False):
        """
        ARGS:
            state : spectrogram of curr state (b * ch * t * f) 
    
        Returns
            Reward for curr state
        """
        
        reward = self.reward_model.get_reward(state)
        if scale:
            reward = reward * 0.1

        return reward 
    
    def get_PESQ_reward(self, next_state):
        """
        Calculate the reward of the current state.
        Reward is defined as the tanh of relative difference between 
        PESQ scores of the noisy and the current enhanced signal.
        """
        length = next_state["est_audio"].size(-1)
        est_audio_list = list(next_state["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(next_state["cl_audio"].cpu().numpy()[:, :length])
        

        z_mask, z = batch_pesq(clean_audio_list,
                            est_audio_list)
        
        pesq_reward = (z_mask * z)

        if self.gpu_id is not None:
            pesq_reward = pesq_reward.to(self.gpu_id)

        return pesq_reward 
    
    def phi(self, imag, real):
        return torch.atan2(imag, real)
    
    def faw(self, t):
        return torch.abs(t - 2*PI * torch.round(t/2*PI))

    def get_angle_reward(self, state):
        enhanced = state['noisy']
        clean = state['clean']

        p_clean = self.phi(clean[:, 0, :, :], clean[:, 1, :, :])
        p_enhanced = self.phi(enhanced[:, 0, :, :], enhanced[:, 1, :, :])

        angle_loss = self.faw(torch.abs(p_clean - p_enhanced)).mean()
        return -angle_loss 


    def get_NISQA_MOS_reward(self, audio, c):
        mos = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            c = c.reshape(-1, 1)
            bs = c.shape[0]
            est_audio = audio/c
            est_audio = est_audio
            est_audio = est_audio.detach().cpu().numpy()
            _dir_ = os.path.join(tmpdirname, 'audios')
            os.makedirs(_dir_, exist_ok=True)
            for i in range(bs):
                save_path = os.path.join(_dir_, f'batch_{i}.wav')
                sf.write(save_path, est_audio[i], 16000)
            
            cmd = f"python ~/NISQA/run_predict.py \
                   --mode predict_dir \
                   --pretrained_model ~/NISQA/weights/nisqa.tar \
                   --data_dir {_dir_} \
                   --num_workers 0 \
                   --bs {bs} \
                   --output_dir {tmpdirname}"

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate() 
            
            with open(os.path.join(tmpdirname, 'NISQA_results.csv'), 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    f, m = line.split(',')[:2]
                    mos.append(float(m))
        
        mos = torch.tensor(mos).to(self.gpu_id)
        mos = mos.reshape(-1, 1)
        return mos

class replay_buffer:
    def __init__(self, max_size, gpu_id=None):
        self.buffer = deque(maxlen=max_size)
        self.gpu_id = gpu_id

    def push(self, state, action, reward, next_state):
        experience = {
            'curr':state,
            'action':action,
            'reward':reward,
            'next':next_state
        }
        self.buffer.append(experience)

    def sample(self, batch_size):
        CURR = {}
        NEXT = {}
        ACTION = [[], []]
        REWARD = []
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        for idx in indices:
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

            action = (torch.FloatTensor(self.buffer[idx]['action'][0]), torch.FloatTensor(self.buffer[idx]['action'][1]))
            ACTION[0].append(action[0])
            ACTION[1].append(action[1])

        ACTION = (torch.stack(ACTION[0]).squeeze(1), torch.stack(ACTION[1]).squeeze(1))
        REWARD = torch.stack(REWARD).unsqueeze(-1)
        CURR = {k:torch.stack(v).squeeze(1) for k, v in CURR.items()}
        NEXT = {k:torch.stack(v).squeeze(1) for k, v in NEXT.items()}
        if self.gpu_id is not None:
            ACTION = (ACTION[0].to(self.gpu_id), 
                      ACTION[1].to(self.gpu_id))
            REWARD = REWARD.to(self.gpu_id)
            CURR = {
                k:v.to(self.gpu_id) for k, v in CURR.items()
                }
            NEXT = {
                k:v.to(self.gpu_id) for k, v in NEXT.items()
                }

        return {
            'curr':CURR, 
            'next':NEXT, 
            'action':ACTION, 
            'reward':REWARD
        }

    def __len__(self):
        return len(self.buffer)
    



# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py

class GaussianStrategy:
    """
    This strategy adds Gaussian noise to the action taken by the deterministic policy.

    Based on the rllab implementation.
    """
    def __init__(self, gpu_id=None, max_sigma=1.0, min_sigma=None,
                 decay_period=1000000):
   
        self._max_sigma = max_sigma
        if min_sigma is None:
            min_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self.gpu_id = gpu_id

    def get_action_from_raw_action(self, action, t=None, **kwargs):
        sigma = (
            self._max_sigma - (self._max_sigma - self._min_sigma) *
            min(1.0, t * 1.0 / self._decay_period)
        )
        
        var = torch.ones(action.shape)
        if self.gpu_id is not None:
            var = var.to(self.gpu_id)

        dist = Normal(action, var * sigma)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        
        #action = torch.clip(
        #    action + rand,
        #    self._action_space.low,
        #    self._action_space.high,
        #)

        #action = action + rand * sigma
        #return action, log_prob
        return action, log_prob



class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.015, max_sigma=0.05, min_sigma=0.05, decay_period=100000, gpu_id=None):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.gpu_id       = gpu_id
        self.state        = {}
        #self.reset(self.action_dim)
        
        
    def reset(self, action_dim):
        b, ch, t, f = action_dim
        key = f"{b}_{ch}_{t}_{f}"
        self.state[key] = torch.ones(action_dim) * self.mu
        if self.gpu_id is not None:
            self.state[key] = self.state[key].to(self.gpu_id)
        
    def evolve_state(self, action):
        action_dim = action.shape
        b, ch, t, f = action_dim
        key = f"{b}_{ch}_{t}_{f}"

        if key not in self.state:
            self.reset(action_dim)
        
        x = self.state[key]   
        rand = torch.randn(action_dim)
        if self.gpu_id is not None:
            rand = rand.to(self.gpu_id)
        
        dx = self.theta * (self.mu - x) + self.sigma * rand
        self.state[key] = x + dx
        return self.state[key]
    
    def get_action(self, action, t=0):
        ou_state_0 = self.evolve_state(action[0])
        ou_state_1 = self.evolve_state(action[1])
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        mag_mask = torch.clip(action[0] + ou_state_0, torch.tensor(0.0).to(self.gpu_id), torch.max(action[0]))
        comp_mask = torch.clip(action[1] + ou_state_1, torch.min(action[1]), torch.max(action[1]))
        return (mag_mask, comp_mask)