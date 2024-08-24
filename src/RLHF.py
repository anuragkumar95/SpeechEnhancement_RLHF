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

torch.manual_seed(0)


class REINFORCE:
    def __init__(self, 
                 loader,
                 batchsize=4, 
                 init_model=None,
                 reward_model=None, 
                 gpu_id=None,
                 beta=0.01, 
                 lmbda=0.1,
                 discount=1.0, 
                 episode_len=1,
                 loss_type=None,
                 reward_type=None, 
                 **params):
        
        self.env = SpeechEnhancementAgent(n_fft=params['env_params'].get("n_fft"),
                                          hop=params['env_params'].get("hop"),
                                          gpu_id=gpu_id,
                                          args=params['env_params'].get("args"),
                                          reward_model=reward_model)
        self.dataloader = loader
        self._iter_ = iter(loader)
        self.bs = batchsize
        self.discount = discount
        self.lmbda = lmbda
        self.gpu_id = gpu_id
        self.rlhf = True
        self.reward_model = reward_model
        if self.reward_model is None:
            self.rlhf = False
        self.beta = beta
        self.t = 0
        self.init_model = init_model
        self.loss_type = loss_type
        self.reward_type = reward_type
    
        if init_model is not None:
            self.init_model = init_model.eval()
        self.train_phase = params['train_phase']
        self.episode_len = episode_len

    def run_episode(self, actor, a_optim):

        #NOTE: We don't want to set actor to train mode due to presence of layer/instance norm layers
        #acting differently in train and eval mode. RL seems to be stable only when actor
        #is still in eval mode
        actor = actor.eval()
        actor.set_evaluation(False)
        actor_sft = self.init_model
        actor_sft = actor_sft.eval()
        actor_sft.set_evaluation(False)

        try:
            batch = next(self._iter_)
        except StopIteration as e:
            self._iter_ = iter(self.dataloader)
            batch = next(self._iter_)

        #Preprocessed batch
        batch = preprocess_batch(batch, gpu_id=self.gpu_id) 
        
        cl_aud, clean, noisy, _ = batch
        noisy = noisy.permute(0, 1, 3, 2)
        clean = clean.permute(0, 1, 3, 2)
       
        #Get logprobs and values for the sampled state
        action, log_probs, _, _ = actor.get_action(noisy)
    
        state = self.env.get_next_state(state=noisy, action=action)
        state['cl_audio'] = cl_aud
        state['clean'] = clean

        sft_action, _, _, _ = self.init_model.get_action(noisy)
        sft_state = self.env.get_next_state(state=noisy, action=sft_action)
        sft_state['cl_audio'] = cl_aud
        sft_state['clean'] = clean
        
        if self.train_phase:
            log_prob = log_probs[0].permute(0, 2, 1) + log_probs[1][:, 0, :, :] + log_probs[1][:, 1, :, :]
        else:
            raise NotImplementedError
        
        with torch.no_grad():
            #Reward model
            rm_score = self.env.get_RLHF_reward(state=state['noisy'].permute(0, 1, 3, 2), 
                                                scale=False)
            
            #PESQ
            mb_pesq = []
            for i in range(self.bs):
                values = compute_metrics(cl_aud[i, ...].detach().cpu().numpy().reshape(-1), 
                                        state['est_audio'][i, ...].detach().cpu().numpy().reshape(-1), 
                                        16000, 
                                        0)

                mb_pesq.append(values[0])

            mb_pesq_sft = []
            for i in range(self.bs):
                values = compute_metrics(cl_aud[i, ...].detach().cpu().numpy().reshape(-1), 
                                        sft_state['est_audio'][i, ...].detach().cpu().numpy().reshape(-1), 
                                        16000, 
                                        0)

                mb_pesq_sft.append(values[0])
            
            mb_pesq = torch.tensor(mb_pesq).to(self.gpu_id)
            mb_pesq = mb_pesq.reshape(-1, 1)
            mb_pesq_sft = torch.tensor(mb_pesq_sft).to(self.gpu_id)
            mb_pesq_sft = mb_pesq_sft.reshape(-1, 1)
            
            #SFT logprobs
            ref_log_probs, _ = self.init_model.get_action_prob(noisy, action)
            ref_log_prob = ref_log_probs[0].permute(0, 2, 1) + ref_log_probs[1][:, 0, :, :] + ref_log_probs[1][:, 1, :, :]

        #Supervised loss
        enhanced = state['noisy']
        enhanced_mag = torch.sqrt(enhanced[:, 0, :, :]**2 + enhanced[:, 1, :, :]**2)
        clean_mag = torch.sqrt(clean[:, 0, :, :]**2 + clean[:, 1, :, :]**2)
        
        mag_loss = (clean_mag - enhanced_mag)**2
        ri_loss = (clean - enhanced) ** 2
        supervised_loss = 0.3 * torch.mean(ri_loss, dim=[1, 2, 3]) + 0.7 * torch.mean(mag_loss, dim=[1, 2])
        supervised_loss = supervised_loss.reshape(-1, 1)

        #KL    
        kl_penalty = torch.mean(log_prob - ref_log_prob, dim=[1, 2]).detach()
        ratio = torch.exp(kl_penalty)
        print(f"KL_ratio:{ratio.mean()}")
        kl_penalty = ((ratio - 1) - kl_penalty)
        kl_penalty = kl_penalty.reshape(-1, 1)

        #Current step reward
        r_t = 0
        if 'rm' in self.reward_type:
            r_t = r_t + rm_score
    
        if 'mse' in self.reward_type:
            r_t = r_t - self.lmbda * supervised_loss
        
        if 'pesq' in self.reward_type:
            r_t = r_t + (mb_pesq - mb_pesq_sft)
            
        if 'kl' in self.reward_type:
            r_t = r_t - self.beta * kl_penalty.detach()

        #Policy gradient loss
        mb_reward = r_t.detach().reshape(-1)
        pg_loss = -torch.einsum("b, bij->bij",mb_reward, log_prob)
        pg_loss = torch.mean(pg_loss, dim=[1, 2])

        #Current step loss
        ovl_loss = 0
        if 'pg' in self.loss_type:
            ovl_loss = ovl_loss + pg_loss

        if 'mse' in self.loss_type:
            ovl_loss = ovl_loss + self.lmbda * supervised_loss

        if 'kl' in self.loss_type:
            ovl_loss = ovl_loss + self.beta * kl_penalty

        ovl_loss = ovl_loss.mean()
        ovl_loss.backward() 

        print(f"pg_loss:{pg_loss.mean()} | MSE :{supervised_loss.mean()} | KL :{kl_penalty.mean()}")
    
        #Update network
        if not torch.isnan(ovl_loss).any():
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            a_optim.step()
            a_optim.zero_grad()

        return (pg_loss.mean(), supervised_loss.mean()), kl_penalty.mean(), (rm_score.mean(), mb_reward.mean()), mb_pesq.mean()  

    
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
                 val_coef=0.02, 
                 en_coef=0.01,
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
        #self.dist = params['env_params'].get("args").out_dist
        self.train_phase = params['train_phase']
        self.t = 0
        self.scale_rewards = scale_rewards
        self.reward_type = reward_type
        self.loss_type = loss_type
        self.warm_up = warm_up_steps
        self.init_model = None
        if init_model is not None:
            self.init_model = init_model.eval()
        self.prev_log_probs = None
        self.val_coef = val_coef
        self.en_coef = en_coef
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
        actor = actor.eval()
        actor.set_evaluation(False)
        self.init_model = self.init_model.eval()
        self.init_model.set_evaluation(True)

        rewards = []
        r_ts = []
        states = []
        logprobs = []
        actions = []
        cleans = []
        ep_kl_penalty = 0
        pesq = 0
        C = []
        with torch.no_grad():
            for _ in range(self.episode_len):

                for _ in range(self.accum_grad):

                    try:
                        batch_rl = next(self._iter_['rl'])
                    except StopIteration as e:
                        self._iter_['rl'] = iter(self.dataloader['rl'])
                        batch_rl = next(self._iter_['rl'])


                    #Preprocessed batch
                    batch_rl = preprocess_batch(batch_rl, gpu_id=self.gpu_id, return_c=True) 
                
                    cl_aud_rl, clean_rl, noisy_rl, _, c_rl = batch_rl
                    noisy_rl = noisy_rl.permute(0, 1, 3, 2)
                    clean_rl = clean_rl.permute(0, 1, 3, 2)
                    bs, ch, t, f = clean_rl.shape
                    
                    action, log_probs, _, _ = actor.get_action(noisy_rl)
                    print(f"action: {len(action)}, {len(action[0])}, {len(action[1])}, {len(action[1][1])}")

                    #if self.model == 'mpsenet':
                        #print(f"Storing actions:{action[0][0].mean()}, {action[0][1].mean()}, {action[1][0].mean()}, {action[1][1][0].mean()}, {action[1][1][1].mean()}")
                        #print(f"Storing actions:{action[0][0].shape}, {action[0][1].shape}, {action[1][0].shape}, {action[1][1][0].shape}, {action[1][1][1].shape}")
            
                    print(f"log_probs:{log_probs[0].mean(), log_probs[1].mean()}")
                    
                    if self.init_model is not None:
                        init_action, _, _, _ = self.init_model.get_action(noisy_rl)
                        ref_log_probs, _ = self.init_model.get_action_prob(noisy_rl, action)
                        #print(f"REF_LOG_PROBS:{ref_log_probs[0].mean()}, {ref_log_probs[1].mean()}")
                        exp_state = self.env.get_next_state(state=noisy_rl, action=init_action, model=self.model)
            
                    state = self.env.get_next_state(state=noisy_rl, action=action, model=self.model)
                    state['cl_audio'] = cl_aud_rl
                    state['clean'] = clean_rl
                    if self.init_model is not None:
                        state['exp_est_audio'] = exp_state['est_audio']

                    #Calculate sft output
                    sft_action, _, _, _ = self.init_model.get_action(noisy_rl)
                    sft_state = self.env.get_next_state(state=noisy_rl, action=sft_action, model=self.model)
                    sft_state['cl_audio'] = cl_aud_rl
                    sft_state['clean'] = clean_rl

                    #Calculate kl_penalty
                    ref_log_prob = None
                    if self.init_model is not None:
                        ref_log_prob = ref_log_probs[0] + ref_log_probs[1][:, 0, :, :].permute(0, 2, 1) + ref_log_probs[1][:, 1, :, :].permute(0, 2, 1)
                        print(f"ref_log_prob:{ref_log_prob.mean()}")
                    log_prob = log_probs[0] + log_probs[1][:, 0, :, :].permute(0, 2, 1) + log_probs[1][:, 1, :, :].permute(0, 2, 1)
                    print(f"log_prob:{log_prob.mean()}")
                    
                    if ref_log_prob is not None:
                        kl_penalty = torch.mean(log_prob - ref_log_prob, dim=[1, 2]).detach()
                        ep_kl_penalty += kl_penalty.mean()
                    else:
                        kl_penalty = None
                    
                    rm_score = torch.tensor(0.0)
                    sft_rm_score = torch.tensor(0.0)
                    #Store reward
                    
                    #if 'rm_nisqa' in self.reward_type:
                    rm_score = self.env.get_NISQA_MOS_reward(audio=state['est_audio'], c=c_rl)
                    sft_rm_score = self.env.get_NISQA_MOS_reward(audio=sft_state['est_audio'], c=c_rl)
                        
                    #elif 'rm' in self.loss_type:
                    #    rm_score = self.env.get_RLHF_reward(state=state['noisy'].permute(0, 1, 3, 2), 
                    #                                   scale=self.scale_rewards)
                    #    sft_rm_score = self.env.get_RLHF_reward(state=sft_state['noisy'].permute(0, 1, 3, 2), 
                    #                                   scale=self.scale_rewards)
                    r_ts.append(rm_score)

                    mb_pesq = []
                    for i in range(self.bs):
                        values = compute_metrics(cl_aud_rl[i, ...].detach().cpu().numpy().reshape(-1), 
                                                state['est_audio'][i, ...].detach().cpu().numpy().reshape(-1), 
                                                16000, 
                                                0)

                        mb_pesq.append(values[0])

                    mb_pesq_sft = []
                    for i in range(self.bs):
                        values = compute_metrics(cl_aud_rl[i, ...].detach().cpu().numpy().reshape(-1), 
                                                sft_state['est_audio'][i, ...].detach().cpu().numpy().reshape(-1), 
                                                16000, 
                                                0)

                        mb_pesq_sft.append(values[0])
                    
                    mb_pesq = torch.tensor(mb_pesq).to(self.gpu_id)
                    pesq += mb_pesq.sum()
                    mb_pesq_sft = torch.tensor(mb_pesq_sft).to(self.gpu_id)
                    
                    kl_penalty = kl_penalty.reshape(-1, 1)
                    mb_pesq = mb_pesq.reshape(-1, 1)
                    mb_pesq_sft = mb_pesq_sft.reshape(-1, 1)

                    #Current step reward
                    r_t = 0
                    if 'rm' in self.reward_type:
                        r_t = r_t + (rm_score - sft_rm_score)
                
                    #if 'mse' in self.reward_type:
                    #    r_t = r_t - self.lmbda * supervised_loss
                    
                    if 'pesq' in self.reward_type:
                        r_t = r_t + (mb_pesq - mb_pesq_sft)
                        
                    if 'kl' in self.reward_type:
                        r_t = r_t - self.beta * kl_penalty

                    #print(f"RM:{r_t.mean()} kl:{kl_penalty.mean()} loss:{supervised_loss.mean()} PESQ:{mb_pesq.mean()}")
                    print(f"RM:{r_t.mean()} kl:{kl_penalty.mean()} PESQ:{mb_pesq.mean()}")
                    
                    #Store trajectory
                    states.append(noisy_rl)
                    cleans.append(clean_rl)
                    rewards.append(r_t)

                    actions.append(action)
                    logprobs.append(log_prob)
                    #logprobs.append(ref_log_prob)

            #Convert collected rewards to target_values and advantages
            rewards = torch.stack(rewards).reshape(bs, -1)
            r_ts = torch.stack(r_ts).reshape(-1)
            target_values = self.get_expected_return(rewards)
            b_target = target_values.reshape(-1)
            rewards = rewards.reshape(-1)
            print(f"REWARDS:{rewards}")
            
            states = torch.stack(states).reshape(-1, ch, t, f)
            cleans = torch.stack(cleans).reshape(-1, ch, t, f)
            
            if self.model == 'cmgan':
                m_actions0 = []
                m_actions1 = []
                c_actions0 = []
             
                for batch in actions:
                    m_actions0.append(batch[0][0])
                    m_actions1.append(batch[0][1])
                    c_actions0.append(batch[1])
                   
                actions = ((torch.stack(m_actions0).reshape(-1, f, t).detach(), 
                            torch.stack(m_actions1).reshape(-1, f, t).detach()),
                            torch.stack(c_actions0).reshape(-1, ch, t, f).detach())
                
            if self.model == 'mpsenet':
                #print(f"len actions:{}")
                #m_actions0 = []
                #m_actions1 = []
                #c_actions0 = []
                #c_actions1 = []
                #c_actions2 = []
                #for batch in actions:
                #    print(batch[0][0].shape, batch[0][1].shape, batch[1][0].shape, batch[1][1][0].shape, batch[1][1][1].shape,)
                #    m_actions0.append(batch[0][0])
                #    m_actions1.append(batch[0][1])
                #    c_actions0.append(batch[1][0])
                #    c_actions1.append(batch[1][1][0])
                #    c_actions2.append(batch[1][1][1])
                
                actions = (
                    (
                        [batch[0][0] for batch in actions],
                        [batch[0][1] for batch in actions]
                    ),
                    (
                        [batch[1][0] for batch in actions],
                        (
                            [batch[1][1][0] for batch in actions],
                            [batch[1][1][1] for batch in actions]
                        )
                    )

                )

                actions = (
                    (
                        torch.stack(actions[0][0]).reshape(-1, 1, t, f).detach(),
                        torch.stack(actions[0][1]).reshape(-1, 1, t, f).detach()
                    ),
                    (
                        torch.stack(actions[1][0]).reshape(-1, 1, t, f).detach(),
                        (
                            torch.stack(actions[1][1][0]).reshape(-1, 1, t, f).detach(),
                            torch.stack(actions[1][1][1]).reshape(-1, 1, t, f).detach()
                        )
                    )
                )
            
            if self.model == 'cmgan':
                logprobs = torch.stack(logprobs).reshape(-1, f, t).detach()
            if self.model == 'mpsenet':
                logprobs = torch.stack(logprobs).reshape(-1, 1, f, t).detach()
            
            ep_kl_penalty = ep_kl_penalty / (self.episode_len * self.accum_grad)
            pesq = pesq / (self.episode_len * self.accum_grad * self.bs)

        print(f"STATES        :{states.shape}")
        print(f"CLEAN         :{cleans.shape}")
        print(f"REWARDS:      :{rewards.shape}")
        print(f"TARGET_VALS   :{b_target.shape}")
        if self.model == 'mpsenet':
            print(f"ACTIONS       :{actions[0][0].shape}, {actions[0][1].shape}, {actions[1][0].shape}, {actions[1][1][0].shape}, {actions[1][1][1].shape}")
        if self.model == 'cmgan':
            print(f"ACTIONS       :{actions[0][0].shape, actions[0][1].shape, actions[1].shape}")
        print(f"LOGPROBS      :{logprobs.shape}")
        #print(f"POLICY RETURNS:{target_values.mean(0)}")

        policy_out = {
            'states':states,
            'cleans':cleans,
            'b_targets':b_target,
            'actions':actions,
            'log_probs':logprobs,
            'target_values':target_values,
            'r_ts':(r_ts, rewards),
            'ep_kl':ep_kl_penalty,
            'pesq':pesq,
            'C':C,
        }
        
        return policy_out

    def train_on_policy(self, policy, actor, optimizers, n_epochs):

        states = policy['states']
        actions = policy['actions']
        logprobs = policy['log_probs']
        target_values = policy['target_values']
        ep_kl_penalty = policy['ep_kl']
        r_ts, reward = policy['r_ts']
        pesq = policy['pesq']
        
        #Start training over the unrolled batch of trajectories
        #Set models to train
        #NOTE: We don't want to set actor to train mode due to presence of layer/instance norm layers
        #acting differently in train and eval mode. PPO seems to be stable only when actor
        #is still in eval mode
        actor = actor.eval()

        a_optim, _ = optimizers
        
        step_clip_loss = 0
        #step_val_loss = 0
        #step_entropy_loss = 0
        pretrain_loss = 0
        step_pg_loss = 0
        #VALUES = torch.zeros(target_values.shape)

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
                batch_pre = preprocess_batch(batch_pre, gpu_id=self.gpu_id, return_c=True) 
            
                cl_aud_pre, clean_pre, noisy_pre, _, c_pre = batch_pre
                noisy_pre = noisy_pre.permute(0, 1, 3, 2)
                clean_pre = clean_pre.permute(0, 1, 3, 2)
            
                #Get mini batch indices
                mb_indx = indices[t:t + self.bs]
                mb_states = states[mb_indx, ...]

                #Get new logprobs and values for the sampled (state, action) pair
                if self.model == 'cmgan':
                    mb_action = ((actions[0][0][mb_indx, ...],
                                  actions[0][1][mb_indx, ...]), 
                                  actions[1][mb_indx, ...])

                if self.model == 'mpsenet':
                    mb_action = ((actions[0][0][mb_indx, ...], 
                                actions[0][1][mb_indx, ...]), 
                                (actions[1][0][mb_indx, ...],
                                (actions[1][1][0][mb_indx, ...], actions[1][1][1][mb_indx, ...])))
                    #print(f"Sampled actions:{mb_action[0][0].mean()}, {mb_action[0][1].mean()}, {mb_action[1][0].mean()}, {mb_action[1][1][0].mean()}, {mb_action[1][1][1].mean()}") 
                    #print(f"Sampled actions:{mb_action[0][0].shape}, {mb_action[0][1].shape}, {mb_action[1][0].shape}, {mb_action[1][1][0].shape}, {mb_action[1][1][1].shape}") 

                log_probs, _ = actor.get_action_prob(mb_states, mb_action)
                ref_log_probs, _ = self.init_model.get_action_prob(mb_states, mb_action)

                if self.train_phase:
                    #entropy = entropies[0].permute(0, 2, 1) + entropies[1][:, 0, :, :] + entropies[1][:, 1, :, :]
                    log_prob = log_probs[0].permute(0, 2, 1) + log_probs[1][:, 0, :, :] + log_probs[1][:, 1, :, :]
                    ref_log_prob = ref_log_probs[0].permute(0, 2, 1) + ref_log_probs[1][:, 0, :, :] + ref_log_probs[1][:, 1, :, :]
                    ref_log_prob = ref_log_prob.detach()
                    old_log_prob = logprobs[mb_indx, ...]#.permute(0, 2, 1)
                    if self.model == 'mpsenet':
                        old_log_prob = old_log_prob.squeeze(1)
                    if self.model == 'cmgan':
                        old_log_prob = old_log_prob.permute(0, 2, 1)

                else:
                    #ignore complex mask, just tune mag mask 
                    raise NotImplementedError
                
                print(f"log_prob:{log_prob.mean(), log_prob.shape}")
                print(f"old_logprob:{old_log_prob.mean(), old_log_prob.shape}")

                #KL Penalty
                kl_logratio = torch.mean(log_prob - ref_log_prob, dim=[1, 2])
                kl_penalty = kl_logratio

                mb_adv = reward[mb_indx, ...].reshape(-1, 1)
                
                #Policy gradient loss

                logratio = torch.mean(log_prob - old_log_prob, dim=[1, 2])
                ratio = torch.exp(logratio).reshape(-1, 1)
                
                print(f"Ratio:{ratio}")
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                pg_loss = torch.max(pg_loss1, pg_loss2)

                mb_act, _, _, _ = actor.get_action(noisy_pre)
                mb_next_state = self.env.get_next_state(state=noisy_pre, action=mb_act, model=self.model)
                mb_enhanced = mb_next_state['noisy']

                mb_enhanced_mag = torch.sqrt(mb_enhanced[:, 0, :, :]**2 + mb_enhanced[:, 1, :, :]**2)
                mb_clean_mag = torch.sqrt(clean_pre[:, 0, :, :]**2 + clean_pre[:, 1, :, :]**2)

                if self.model == 'mpsenet':
                    mb_enhanced_mag = mb_enhanced_mag.permute(0, 2, 1)
                    mb_enhanced = mb_enhanced.permute(0, 1, 3, 2)

                supervised_loss = ((clean_pre - mb_enhanced) ** 2).mean() + ((mb_clean_mag - mb_enhanced_mag)**2).mean()
                
                pretrain_loss += supervised_loss.detach()

                clip_loss = 0
                if 'pg' in self.loss_type:
                    clip_loss = clip_loss + pg_loss.reshape(-1, 1)

                if 'mse' in self.loss_type:
                    clip_loss = clip_loss + self.lmbda * supervised_loss.reshape(-1, 1)

                if 'kl' in self.loss_type:
                    clip_loss = clip_loss + self.beta * kl_penalty.reshape(-1, 1)
                
                clip_loss = clip_loss.mean()

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
               (target_values.mean(), ep_kl_penalty, r_ts.mean(), reward.mean()), pesq  

    