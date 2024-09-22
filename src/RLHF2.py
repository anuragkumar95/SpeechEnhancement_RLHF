# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

#from model.CMGAN.actor import TSCNet
#from model.critic import QNet
#import os
from data.dataset import load_data
import torch.nn.functional as F
import torch
from utils import preprocess_batch, power_compress, power_uncompress, batch_pesq, copy_weights, freeze_layers, original_pesq
#import logging
from torchinfo import summary
#import argparse
import wandb
#import psutil
import numpy as np
#import traceback
from speech_enh_env import  SpeechEnhancementAgent, GaussianStrategy
import torch
import wandb
#import copy
from torch.distributions import Normal

from compute_metrics import compute_metrics

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

torch.manual_seed(123)


class PPO:
    """
    Base class for PPO policy gradient method.
    """
    def __init__(self, 
                 loader,
                 init_model, 
                 reward_model, 
                 gpu_id=None,
                 batchsize=4, 
                 run_steps=1, 
                 beta=0.2,
                 eps=0.01, 
                 lmbda=0,  
                 discount=1.0, 
                 accum_grad=1,
                 loss_type=None,
                 reward_type=None,
                 scale_rewards=False, 
                 warm_up_steps=30, 
                 model='cmgan',
                 **params):
        
        self.env = SpeechEnhancementAgent(n_fft=params['env_params'].get("n_fft"),
                                          hop=params['env_params'].get("hop"),
                                          gpu_id=gpu_id,
                                          args=params['env_params'].get("args"),
                                          reward_model=reward_model)
        self.dataloader = loader
        self._iter_ = {'pre':iter(loader['pre']), 'rl':iter(loader['rl'])}
        self.bs = batchsize
        self.discount = discount
        self.beta = beta
        self.lmbda = lmbda
        self.eps = eps
        self.gpu_id = gpu_id
        self.accum_grad = accum_grad
        self.rlhf = True
        self.train_phase = params['train_phase']
        self.t = 0
        self.scale_rewards = scale_rewards
        self.reward_type = reward_type
        self.loss_type = loss_type
        self.warm_up = warm_up_steps
        self.init_model = init_model
        self.episode_len = run_steps
        self.model = model
        print(f"RLHF:{self.rlhf}")


    def run_episode(self, actor, optimizer, n_epochs=3):
        #Start PPO
        policy = self.unroll_policy(actor)
        self.t += 1
        return self.train_on_policy(policy, actor, optimizer, n_epochs)
       
    def get_expected_return(self, rewards):
        """
        Expects rewards to be a torch tensor.
        """
        G = torch.zeros(rewards.shape).to(self.gpu_id)
        for i in range(self.episode_len):
            #Base case: G(T) = r(T)
            #Recursive: G(t) = r(t) + G(t+1)*DISCOUNT
            r_t = rewards[:, self.episode_len - i - 1].detach()
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
                a_t = g_t - critic(states[self.episode_len - i - 1]).detach()
            else:
                a_t = g_t - critic(states[self.episode_len - i - 1]).detach() + self.discount * critic(states[self.episode_len - i]).detach()
            A[:, self.episode_len - i - 1] = a_t.reshape(-1)
        return A
    

    def unroll_policy(self, actor):
        #Set models to eval
        if self.model == 'cmgan':
            self.init_model.eval()
            actor.eval()
            
        if self.model == 'metricgan':
            self.init_model.train()
            actor.train()
            
        self.init_model.evaluation = True
        actor.evaluation = False

        states = []
        logprobs = []
        actions = []
        cleans = []
        KL = []
        C = []
        rl_res = []
        sft_res = []
        cl_audios = []
        actions = []
        pesq = 0
        ep_kl_penalty = 0

        with torch.no_grad():
            for _ in range(self.episode_len):

                for _ in range(self.accum_grad):

                    try:
                        batch_rl = next(self._iter_['rl'])
                    except StopIteration as e:
                        self._iter_['rl'] = iter(self.dataloader['rl'])
                        batch_rl = next(self._iter_['rl'])

                    #Preprocessed batch
                    batch_rl = preprocess_batch(batch_rl, 
                                                n_fft=self.env.n_fft, 
                                                hop=self.env.hop, 
                                                gpu_id=self.gpu_id, 
                                                return_c=True,
                                                model=self.model) 
                
                    cl_aud_rl, clean_rl, noisy_rl, _, c_rl = batch_rl
                    noisy_rl = noisy_rl.permute(0, 1, 3, 2)
                    clean_rl = clean_rl.permute(0, 1, 3, 2) 
                    _, ch, t, f = clean_rl.shape

                    if torch.isnan(noisy_rl.mean()) or torch.isnan(clean_rl.mean()):
                        continue 
                    if torch.isinf(noisy_rl.mean()) or torch.isinf(clean_rl.mean()):
                        continue 
                    
                    sft_next_state, _, _, = self.init_model.get_action(noisy_rl)
                    next_state, log_probs, _ = actor.get_action(noisy_rl)
                    ref_log_probs = self.init_model.get_action_prob(noisy_rl, next_state)

                    sft_audio = self.env.get_audio(sft_next_state)
                    est_audio = self.env.get_audio(next_state)
                    
                    #Calculate kl_penalty
                    ref_log_prob = ref_log_probs[0] + ref_log_probs[1]
                    log_prob = log_probs[0] + log_probs[1]
                    print(f"LOGPROB:{log_prob.shape}")
                    kl_penalty = torch.mean(log_prob - ref_log_prob, dim=[1, 2, 3]).detach()
                    ep_kl_penalty += kl_penalty.mean()
                 
                    #Store trajectory
                    for i in range(next_state[0].shape[0]):
                        states.append(noisy_rl[i, ...])
                        cleans.append(clean_rl[i, ...])
                        logprobs.append(log_prob[i])
                        actions.append((next_state[0][i], next_state[1][i]))
                        rl_res.append(est_audio[i, ...])
                        sft_res.append(sft_audio[i, ...])
                        cl_audios.append(cl_aud_rl[i, ...])
                        C.append(c_rl[i, ...])
                        KL.append(kl_penalty[i])

            #Get MOS rewards
            if 'rm' in self.reward_type:
                rm_score = self.env.get_NISQA_MOS_reward(audios=rl_res, Cs=C)
                sft_rm_score = self.env.get_NISQA_MOS_reward(audios=sft_res, Cs=C)
                rewards = (rm_score - sft_rm_score).reshape(-1)
                
            #Get PESQ reward
            mb_pesq = []
            mb_pesq_sft = []
            for mb_aud, mb_est_aud, mb_est_sft_aud in zip(cl_audios, rl_res, sft_res):
                values = compute_metrics(mb_aud.detach().cpu().numpy().reshape(-1), 
                                        mb_est_aud.detach().cpu().numpy().reshape(-1), 
                                        16000, 
                                        0)
                mb_pesq.append(values[0])
                pesq += values[0]

                if 'pesq' in self.reward_type:
                    values = compute_metrics(mb_aud.detach().cpu().numpy().reshape(-1), 
                                            mb_est_sft_aud.detach().cpu().numpy().reshape(-1), 
                                            16000, 
                                            0)
                    
                    mb_pesq_sft.append(values[0])

            if 'pesq' in self.reward_type:
                mb_pesq = torch.tensor(mb_pesq).to(self.gpu_id)
                mb_pesq_sft = torch.tensor(mb_pesq_sft).to(self.gpu_id)
                rewards = rewards + (mb_pesq - mb_pesq_sft).reshape(-1)

            if 'kl' in self.reward_type:
                KL = torch.stack(KL).reshape(-1).to(self.gpu_id)
                print(rewards.shape, KL.shape)
                rewards = rewards - self.beta * KL
            
            print(f"REWARDS:{rewards}")

            rm_score = rm_score.reshape(-1)        
            states = torch.stack(states).reshape(-1, ch, t, f)
            cleans = torch.stack(cleans).reshape(-1, ch, t, f)
            logprobs = torch.stack(logprobs).reshape(-1, f, t).detach() 
               
                  
            ep_kl_penalty = ep_kl_penalty / (self.episode_len * self.accum_grad)
            pesq = pesq / (self.episode_len * self.accum_grad * self.bs)

        print(f"STATES        :{states.shape}")
        print(f"CLEAN         :{cleans.shape}")
        print(f"REWARDS:      :{rewards.shape}")
        print(f"ACTIONS       :{len(actions)}, {actions[0][0].shape, actions[0][1].shape}")
        print(f"LOGPROBS      :{logprobs.shape}")

        policy_out = {
            'states':states,
            'cleans':cleans,
            'actions':actions,
            'log_probs':logprobs,
            'r_ts':(rm_score, rewards),
            'ep_kl':ep_kl_penalty,
            'pesq':pesq,
            'C':C,
        }
        
        return policy_out

    def train_on_policy(self, policy, actor, optimizers, n_epochs):

        states = policy['states']
        actions = policy['actions']
        logprobs = policy['log_probs']
        ep_kl_penalty = policy['ep_kl']
        r_ts, reward = policy['r_ts']
        pesq = policy['pesq']
        
        #Start training over the unrolled batch of trajectories
        #Set models to train
        
        #NOTE: We don't want to set actor to train mode due to presence of layer/instance norm layers
        #acting differently in train and eval mode. PPO seems to be stable only when actor
        #is still in eval mode
        actor.eval()

        a_optim, _ = optimizers
        
        step_clip_loss = 0
        pretrain_loss = 0
        step_pg_loss = 0

        for _ in range(n_epochs):
            indices = [t for t in range(states.shape[0])]
            np.random.shuffle(indices)
            for t in range(0, len(indices), self.bs):

                try:
                    batch_pre = next(self._iter_['pre'])
                except StopIteration as e:
                    self._iter_['pre'] = iter(self.dataloader['pre'])
                    batch_pre = next(self._iter_['pre'])
            
                #Preprocessed batch
                batch_pre = preprocess_batch(batch_pre, 
                                                n_fft=self.env.n_fft, 
                                                hop=self.env.hop, 
                                                gpu_id=self.gpu_id, 
                                                return_c=True,
                                                model=self.model) 
            
                _, clean_pre, noisy_pre, _, c_pre = batch_pre
                noisy_pre = noisy_pre.permute(0, 1, 3, 2)
                clean_pre = clean_pre.permute(0, 1, 3, 2)
                
                if torch.isnan(noisy_pre.mean()) or torch.isnan(clean_pre.mean()):
                    continue 
                if torch.isinf(noisy_pre.mean()) or torch.isinf(clean_pre.mean()):
                    continue 
            
                #Get mini batch indices
                mb_indx = indices[t:t + self.bs]
                mb_states = states[mb_indx, ...]
                mb_actions = [actions[i] for i in mb_indx]
                mb_actions = (torch.stack([act[0] for act in mb_actions]), 
                                torch.stack([act[1] for act in mb_actions]))
                
                #Get new logprobs and values for the sampled (state, action) pair
                log_probs = actor.get_action_prob(mb_states, mb_actions)
                ref_log_probs = self.init_model.get_action_prob(mb_states, mb_actions)

                if self.train_phase:
                    log_prob = log_probs[0]+log_probs[1]
                    ref_log_prob = (ref_log_probs[0] + ref_log_probs[1]).detach()
                    old_log_prob = logprobs[mb_indx, ...].permute(0, 2, 1).unsqueeze(1)

                else:
                    #ignore complex mask, just tune mag mask 
                    raise NotImplementedError
                    
                #KL Penalty
                kl_logratio = torch.mean(log_prob - ref_log_prob, dim=[1, 2, 3])
                kl_penalty = kl_logratio

                mb_adv = reward[mb_indx, ...].reshape(-1, 1)
                
                print(f"REWARD:{mb_adv.shape}, logprob:{log_prob.shape}, old_logprob:{old_log_prob.shape}, ref_logprob:{ref_log_prob.shape}")

                #Policy gradient loss
                logratio = torch.mean(log_prob - old_log_prob, dim=[1, 2, 3])
                ratio = torch.exp(logratio).reshape(-1, 1)


                
                print(f"Ratio:{ratio}")
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                pg_loss = torch.max(pg_loss1, pg_loss2)

                #Pretrain loss
                mb_enhanced, _, _ = actor.get_action(noisy_pre)
                mb_enhanced_mag = torch.sqrt(mb_enhanced[0]**2 + mb_enhanced[1]**2)
                mb_clean_mag = torch.sqrt(clean_pre[:, 0, :, :]**2 + clean_pre[:, 1, :, :]**2)
                mag_loss = ((mb_clean_mag - mb_enhanced_mag)**2).mean() 
                mb_enhanced = torch.cat(mb_enhanced, dim=1)
                ri_loss = ((clean_pre - mb_enhanced) ** 2).mean()
                supervised_loss = 0.7*mag_loss + 0.3*ri_loss

                clip_loss = 0
                if 'pg' in self.loss_type:
                    clip_loss = clip_loss + pg_loss.reshape(-1, 1)

                if 'mse' in self.loss_type:
                    clip_loss = clip_loss + self.lmbda * supervised_loss.reshape(-1, 1)

                if 'kl' in self.loss_type:
                    clip_loss = clip_loss + self.beta * kl_penalty.reshape(-1, 1)
                
                clip_loss = clip_loss.mean()
                pretrain_loss += supervised_loss.detach().mean()

                print(f"clip_loss:{clip_loss.item()} | pg_loss:{pg_loss.mean()} | mse: {supervised_loss.mean()} | kl: {kl_penalty.mean()}")
                wandb.log({
                    'ratio':ratio.mean(),
                    'pg_loss1':pg_loss1.mean(),
                    'pg_loss2':pg_loss2.mean(),
                })

                clip_loss = clip_loss / self.accum_grad
                clip_loss.backward()
        
                #Update network
                if not (torch.isnan(clip_loss).any() or torch.isinf(clip_loss).any()) and (self.t % self.accum_grad == 0):
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    a_optim.step()
                    a_optim.zero_grad()
                
                step_clip_loss += clip_loss.mean()
                step_pg_loss += pg_loss.mean()     
                self.t += 1

            step_clip_loss = step_clip_loss / (n_epochs * self.episode_len)
            step_pg_loss = step_pg_loss / (n_epochs * self.episode_len)
            pretrain_loss = pretrain_loss / (n_epochs * self.episode_len)
            
        return (step_clip_loss, step_pg_loss, pretrain_loss), \
               (ep_kl_penalty, r_ts.mean(), reward.mean()), pesq  

    