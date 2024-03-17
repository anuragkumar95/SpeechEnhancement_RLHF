# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.actor import TSCNet
from model.critic import QNet
import os
from data.dataset import load_data
import torch.nn.functional as F
import torch
from utils import preprocess_batch, power_compress, power_uncompress, batch_pesq, copy_weights, freeze_layers, original_pesq
import logging
from torchinfo import summary
import argparse
import wandb
import psutil
import numpy as np
import traceback
from speech_enh_env import  SpeechEnhancementAgent, GaussianStrategy
import torch
import wandb

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class REINFORCE:
    def __init__(self, 
                 init_model, 
                 reward_model, 
                 gpu_id=None, 
                 beta=0.01, 
                 discount=1.0, 
                 episode_len=1,
                 **params):
        
        self.env = SpeechEnhancementAgent(n_fft=params['env_params'].get("n_fft"),
                                          hop=params['env_params'].get("hop"),
                                          gpu_id=gpu_id,
                                          args=params['env_params'].get("args"),
                                          reward_model=reward_model)
        self.discount = discount
        self.gpu_id = gpu_id
        self.rlhf = True
        self.reward_model = reward_model
        self.expert = init_model
        if self.reward_model is None:
            self.rlhf = False
        self.beta = beta
        self.gaussian_noise = GaussianStrategy(gpu_id=gpu_id)
        self.t = 0
        self.dist = params['env_params'].get("args").out_dist
        self.train_phase = params['train_phase']
        self.episode_len = episode_len
        self._r_mavg = 0

    def get_expected_reward(self, rewards):
        """
        Expects rewards to be a numpy array.
        """
        G_t = torch.zeros(rewards.shape).to(self.gpu_id)
        episode_len = rewards.shape[1]
        for i in range(episode_len):
            #Base case: G(T) = r(T)
            #Recursive: G(t) = r(t) + G(t+1)*DISCOUNT
            r_t = rewards[:, episode_len - i - 1]
            if i == 0:
                G_t[:, episode_len - i - 1] = r_t
            else:
                G_t[:, episode_len - i - 1] = r_t + G_t[:, episode_len - i] * self.discount
        return G_t

    def run_one_step_episode(self, batch, model, optimizer):
        """
        Runs an epoch using REINFORCE.
        """
        #Preprocessed batch
        cl_aud, _, noisy, _ = batch

        #Forward pass through model to get the action(mask)
        noisy = noisy.permute(0, 1, 3, 2)
        action, log_probs, _ = model.get_action(noisy)

        #Forward pass through expert model
        exp_action, exp_log_probs, _ = self.expert.get_action(noisy)

        kl_penalty = 0
        
        if self.dist == False:
            #Add gaussian noise
            m_action, log_prob = self.gaussian_noise.get_action_from_raw_action(action[0], t=self.t)
            action = (m_action, action[-1])

        else:
            if self.train_phase:
                #finetune both mag and phase
                log_prob = log_probs[0] + log_probs[1][:, 0, :, :].permute(0, 2, 1) + log_probs[1][:, 1, :, :].permute(0, 2, 1)
                exp_log_prob = exp_log_probs[0] + exp_log_probs[1][:, 0, :, :].permute(0, 2, 1) + exp_log_probs[1][:, 1, :, :].permute(0, 2, 1)
                a_t = action
                #kl divergence term
                kl_penalty = self.beta * torch.mean((log_prob - exp_log_prob), dim=[1, 2]).reshape(-1, 1)
                print(f"KL:{kl_penalty}, {kl_penalty.shape}")
            else:  
                #ignore complex mask, just tune mag mask 
                log_prob = log_probs[0]
                a_t = (action[0], exp_action[-1])
            
        #Apply mask to get the next state
        next_state = self.env.get_next_state(state=noisy, action=a_t)
        next_state['cl_audio'] = cl_aud

        #Get enhanced output
        enhanced = torch.cat([next_state['est_real'], next_state['est_imag']], dim=1).detach()
        self.t += 1
        
        #Get the reward
        if not self.rlhf:
            #Apply exp_mask to get next state
            exp_next_state = self.env.get_next_state(state=noisy, action=exp_action)
            next_state['exp_est_audio'] = exp_next_state['est_audio']
            G = self.env.get_PESQ_reward(next_state)
        else:
            r_t = self.env.get_RLHF_reward(inp=noisy, out=enhanced)
            #Baseline is moving average of rewards seen so far
            self._r_mavg = (self._r_mavg * (self.t - 1) + r_t.mean() ) / self.t
            G = r_t - self._r_mavg - kl_penalty
        
        G = G.reshape(-1, 1)
        print(f"G:{G.mean().item()}, {G.shape}")

        loss = []
        alpha = 1
        for i in range(G.shape[0]):
            loss.append(alpha * -G[i, ...] * log_prob[i, ...] )
        loss = torch.stack(loss).mean()

        print(f"M_LPROB:{log_prob.mean()}")
        print(f"LOSS:{loss.item()}")

        #Update network
        if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)                                                                                
            optimizer.step()

        return loss, (G.mean(), r_t.mean()), enhanced
    
    def run_n_step_episode(self, batch, model, optimizer):
        curr = None
        episode_loss = 0
        episode_return = 0
        for step in range(self.episode_len):
            if step > 0:
                assert curr != None, "Curr is None, check your current update."
                cl_aud, clean, _, labels = batch
                batch = (cl_aud, clean, curr, labels)

            step_loss, step_return, enhanced = self.run_one_step_episode(batch, model, optimizer)
            curr = enhanced
            episode_loss += step_loss.item()
            episode_return += step_return.item() 

        episode_loss = episode_loss / self.episode_len
        episode_return = episode_return / self.episode_len
        return episode_loss, episode_return
    
    def run_episode(self, batch, model, optimizer):
        if self.episode_len == 1:
            loss, reward, _ = self.run_one_step_episode(batch, model, optimizer)
            return loss, reward
        else:
            return self.run_n_step_episode(batch, model, optimizer)


class PPO:
    """
    Base class for PPO ploicy gradient method.
    """
    def __init__(self, 
                 init_model, 
                 reward_model, 
                 gpu_id=None, 
                 run_steps=1, 
                 beta=0.2, 
                 val_coef=0.02, 
                 en_coef=0.01, 
                 discount=1.0, 
                 accum_grad=1,
                 warm_up_steps=1000, 
                 **params):
        
        self.env = SpeechEnhancementAgent(n_fft=params['env_params'].get("n_fft"),
                                          hop=params['env_params'].get("hop"),
                                          gpu_id=gpu_id,
                                          args=params['env_params'].get("args"),
                                          reward_model=reward_model)
        
        self.discount = discount
        self.beta = beta
        self.gpu_id = gpu_id
        self.accum_grad = accum_grad
        self.rlhf = True
        if reward_model is None:
            self.rlhf = False
        self.dist = params['env_params'].get("args").out_dist
        self.train_phase = params['train_phase']
        self.t = 0
        self.warm_up = warm_up_steps
        self.init_model = init_model
        self.prev_log_probs = None
        #self.prev_log_probs = {'noisy':None, 'clean':None}
        self.val_coef = val_coef
        self.en_coef = en_coef
        self.episode_len = run_steps
        self._r_mean = 0
        self._r2_mean = 0
        print(f"RLHF:{self.rlhf}")


    def run_episode(self, batch, actor, critic, optimizers):
        if self.episode_len <= 1:
            return self.run_one_step_episode(batch, actor, critic, optimizers[0])
        else:
            print(f"self.t : {self.t}")
            if self.t < self.warm_up:
                print(f"Training critic")
                return self.train_critic(batch, actor, critic, optimizers[1])
            else:
                return self.run_n_step_episode(batch, actor, critic, optimizers[0])

    def run_one_step_episode(self, batch, actor, critic, optimizer):
        """
        Imagine the episode N --> C --> Terminal
        So for target values, we consider Noisy --> Clean --> Terminal
        and for current iteration values we consider Noisy --> Enhanced --> Terminal
        """
        pass
    
    def get_expected_return(self, rewards):
        """
        Expects rewards to be a torch tensor.
        """
        G = torch.zeros(rewards.shape).to(self.gpu_id)
        for i in range(self.episode_len):
            #Base case: G(T) = r(T)
            #Recursive: G(t) = r(t) + G(t+1)*DISCOUNT
            r_t = rewards[:, self.episode_len - i - 1]
            if i == 0:
                G[:, self.episode_len - i - 1] = r_t
            else:
                G[:, self.episode_len - i - 1] = r_t + G[:, self.episode_len - i] * self.discount
        return G
    
    def get_advantages(self, returns, states, critic): 
        A = torch.zeros(returns.shape).to(self.gpu_id)
        #Base case: A(t) = G(t) - V(S(t)) ; t == T-1
        #Recursive case: A(t) = G(t) - V(S(t)) + discount * V(S(t+1))
        for i in range(self.episode_len):
            g_t = returns[:, self.episode_len - i - 1].reshape(-1, 1)
            if i == 0:
                a_t = g_t - critic(states[self.episode_len - i - 1])
            else:
                a_t = g_t - critic(states[self.episode_len - i - 1]) + self.discount * critic(states[self.episode_len - i])
            A[:, self.episode_len - i - 1] = a_t.reshape(-1)
        return A
    

    def train_critic(self, batch, actor, critic, optimizer):
        #Preprocessed batch
        cl_aud, clean, noisy, _ = batch
        noisy = noisy.permute(0, 1, 3, 2)
        clean = clean.permute(0, 1, 3, 2)
        bs = clean.shape[0]
        critic.eval()
        actor.eval()
        ep_kl_penalty = 0
        
        #Calculate target values and advantages
        with torch.no_grad():
            curr = noisy
            rewards = []
            states = []
            for _ in range(self.episode_len):
                #Unroll policy for n steps and store rewards.
                action, log_probs, _ = actor.get_action(curr)
                init_action, ref_log_probs, _ = self.init_model.get_action(curr)

                state = self.env.get_next_state(state=curr, action=action)
                exp_state = self.env.get_next_state(state=curr, action=init_action)
                state['cl_audio'] = cl_aud
                state['exp_est_audio'] = exp_state['est_audio']

                #Calculate kl_penalty
                ref_log_prob = ref_log_probs[0] + ref_log_probs[1][:, 0, :, :].permute(0, 2, 1) + ref_log_probs[1][:, 1, :, :].permute(0, 2, 1)
                log_prob = log_probs[0] + log_probs[1][:, 0, :, :].permute(0, 2, 1) + log_probs[1][:, 1, :, :].permute(0, 2, 1)
                kl_penalty = torch.mean(log_prob - ref_log_prob, dim=[1, 2])
                ep_kl_penalty += kl_penalty
                
                #Store reward
                if self.rlhf:
                    r_t = self.env.get_RLHF_reward(inp=curr, out=state['noisy'].permute(0, 1, 3, 2))    
                else:
                    r_t = self.env.get_PESQ_reward(state)
                
                r_t = r_t - self.beta * kl_penalty
                rewards.append(r_t)

                #Store state
                states.append(curr)
                curr = state['noisy']

            #Convert collected rewards to target_values and advantages
            rewards = torch.stack(rewards).reshape(bs, -1)
            target_values = self.get_expected_return(rewards)

        critic.train()

        for t in range(len(states)):
            values = critic(states[t])
            #value_loss
            v_loss = ((target_values[:, t] - values) ** 2).mean()

            optimizer.zero_grad()
            v_loss.backward()

            #Update network
            if not (torch.isnan(v_loss).any() or torch.isinf(v_loss).any()) and (self.t % self.accum_grad == 0):
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                optimizer.step()

            self.t += 1
            wandb.log({
                't':self.t,
                'v_loss':v_loss.item(),
                'cummulative_G_t':target_values.mean().item(),
                'critic_values':values.mean().item()
            })

        return None, None

    
    def run_n_step_episode(self, batch, actor, critic, optimizer):
        """
        Imagine the episode N --> C --> Terminal
        So for target values, we consider Noisy --> Clean --> Terminal
        and for current iteration values we consider Noisy --> Enhanced --> Terminal
        """
        #Preprocessed batch
        cl_aud, clean, noisy, _ = batch
        noisy = noisy.permute(0, 1, 3, 2)
        clean = clean.permute(0, 1, 3, 2)
        bs = clean.shape[0]
        critic.eval()
        actor.eval()
        ep_kl_penalty = 0
        
        #Calculate target values and advantages
        with torch.no_grad():
            curr = noisy
            rewards = []
            states = []
            for _ in range(self.episode_len):
                #Unroll policy for n steps and store rewards.
                action, log_probs, _ = actor.get_action(curr)
                init_action, ref_log_probs, _ = self.init_model.get_action(curr)

                state = self.env.get_next_state(state=curr, action=action)
                exp_state = self.env.get_next_state(state=curr, action=init_action)
                state['cl_audio'] = cl_aud
                state['exp_est_audio'] = exp_state['est_audio']

                #Calculate kl_penalty
                ref_log_prob = ref_log_probs[0] + ref_log_probs[1][:, 0, :, :].permute(0, 2, 1) + ref_log_probs[1][:, 1, :, :].permute(0, 2, 1)
                log_prob = log_probs[0] + log_probs[1][:, 0, :, :].permute(0, 2, 1) + log_probs[1][:, 1, :, :].permute(0, 2, 1)
                kl_penalty = torch.mean(log_prob - ref_log_prob, dim=[1, 2])
                ep_kl_penalty += kl_penalty
                
                #Store reward
                if self.rlhf:
                    r_t = self.env.get_RLHF_reward(inp=curr, out=state['noisy'].permute(0, 1, 3, 2))    
                else:
                    r_t = self.env.get_PESQ_reward(state)
                print(f"R:{r_t.reshape(-1)} | KL:{kl_penalty.reshape(-1)}")
                r_t = r_t - self.beta * kl_penalty
                print(f"G:{r_t.reshape(-1)}")
                rewards.append(r_t)

                #Store state
                states.append(curr)
                curr = state['noisy']

            #Convert collected rewards to target_values and advantages
            rewards = torch.stack(rewards).reshape(bs, -1)
            target_values = self.get_expected_return(rewards)
            advantages = self.get_advantages(target_values, states, critic)
            ep_kl_penalty = ep_kl_penalty / self.episode_len
        print(f"Policy returns:{target_values.mean(0)}")

        #Start training over the unrolled batch of trajectories
        actor.train()
        critic.train()
        step_clip_loss = 0
        step_val_loss = 0
        step_entropy_loss = 0
        VALUES = torch.zeros(target_values.shape)
        for t in range(len(states)):
            #Forward pass through model to get the action(mask)
            action, log_probs, entropies = actor.get_action(states[t])
            values = critic(states[t])
            _, exp_log_probs, _ = self.init_model.get_action(states[t])
            VALUES[:, t] = values.reshape(-1)
                
            #Get previous model log_probs 
            if self.prev_log_probs == None:
                self.prev_log_probs = (exp_log_probs[0].detach(), exp_log_probs[1].detach())
            
            if self.train_phase:
                entropy = entropies[0] + entropies[1][:, 0, :, :].permute(0, 2, 1) + entropies[1][:, 1, :, :].permute(0, 2, 1)
                log_prob = log_probs[0] + log_probs[1][:, 0, :, :].permute(0, 2, 1) + log_probs[1][:, 1, :, :].permute(0, 2, 1)
                old_log_prob = self.prev_log_probs[0] + \
                               self.prev_log_probs[1][:, 0, :, :].permute(0, 2, 1) + \
                               self.prev_log_probs[1][:, 1, :, :].permute(0, 2, 1)
                
            else:
                #ignore complex mask, just tune mag mask 
                entropy = entropies[0]
                log_prob, old_log_prob = log_probs[0], self.prev_log_probs[0]
            
            logratio = torch.mean(log_prob - old_log_prob, dim=[1, 2]) 
            ratio = torch.exp(logratio)
            exp_log_prob = exp_log_probs[0] + exp_log_probs[1][:, 0, :, :].permute(0, 2, 1) + exp_log_probs[1][:, 1, :, :].permute(0, 2, 1)
            kl_penalty = torch.mean((log_prob - exp_log_prob), dim=[1, 2]).reshape(-1, 1).detach()

            #Policy loss
            pg_loss1 = -advantages[:, t] * ratio
            pg_loss2 = -advantages[:, t] * torch.clamp(ratio, 1 - self.beta, 1 + self.beta)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            #value_loss
            v_loss = 0.5 * ((target_values[:, t] - values) ** 2).mean()

            #Entropy loss
            entropy_loss = entropy.mean()

            clip_loss = pg_loss - (self.en_coef * entropy_loss) + (self.val_coef * v_loss)

            optimizer.zero_grad()
            clip_loss.backward()
            #Update network
            if not (torch.isnan(clip_loss).any() or torch.isinf(clip_loss).any()) and (self.t % self.accum_grad == 0):
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()

            self.prev_log_probs = (log_probs[0].detach(), log_probs[1].detach())

            step_clip_loss += clip_loss.item()
            step_val_loss += v_loss.item()
            step_entropy_loss += entropy_loss.item()        
            self.t += 1
        
        print(f"Values:{VALUES.mean(0)}")

        step_clip_loss = step_clip_loss / self.episode_len
        step_val_loss = step_val_loss / self.episode_len
        step_entropy_loss = step_entropy_loss / self.episode_len
                    
        return (step_clip_loss, step_val_loss, step_entropy_loss), (target_values.mean(), VALUES.mean(), ep_kl_penalty.mean())

            

                

            

        