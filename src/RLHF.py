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
import copy
from torch.distributions import Normal

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

torch.manual_seed(123)

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
        cl_aud, clean, noisy, _ = batch

        #Forward pass through model to get the action(mask)
        noisy = noisy.permute(0, 1, 3, 2)
        action, log_probs, _, params = model.get_action(noisy)
        (p_mu, p_var), (pc_mu, pc_var) = params

        #Forward pass through expert model
        exp_action, _, _, ref_params = self.expert.get_action(noisy)
        exp_log_probs, _ = self.expert.get_action_prob(noisy, action)
        (r_mu, r_var), (rc_mu, rc_var) = ref_params

        print(f"NEW_PARAMS: MU: {p_mu.min(), p_mu.max(), p_mu.mean()} | VAR: {p_var.min(), p_var.max(), p_var.mean()}")
        print(f"NEW_PARAMS: C_MU: {pc_mu.min(), pc_mu.max(), pc_mu.mean()} | C_VAR: {pc_var.min(), pc_var.max(), pc_var.mean()}")
        print(f"REF_PARAMS: MU: {r_mu.min(), r_mu.max(), r_mu.mean()} | VAR: {r_var.min(), r_var.min().max(), r_var.min().mean()}")
        print(f"REF_PARAMS: C_MU: {rc_mu.min(), rc_mu.max(), rc_mu.mean()} | C_VAR: {rc_var.min(), rc_var.min().max(), rc_var.min().mean()}")
                       

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
                #kl_penalty = self.beta * torch.mean((log_prob - exp_log_prob), dim=[1, 2]).reshape(-1, 1)
                print(f"new_logprob:{log_prob.mean()}")
                print(f"ref_logprob:{exp_log_prob.mean()}")
                kl_penalty = torch.mean(log_prob - exp_log_prob, dim=[1, 2]).detach()
                #ratio = torch.exp(kl_penalty)
               
                #with torch.no_grad():
                    #kl_penalty = ((ratio - 1) - kl_penalty).mean().detach()
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
            next_state['clean'] = clean
            r_t = self.env.get_PESQ_reward(next_state)
        else:
            r_t = self.env.get_RLHF_reward(inp=noisy, out=enhanced)

        #Baseline is moving average of rewards seen so far
        self._r_mavg = (self._r_mavg * (self.t - 1) + r_t.mean() ) / self.t
        G = r_t - self._r_mavg - self.beta * kl_penalty
        
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

        return loss, (G.mean(), r_t.mean(), kl_penalty), enhanced
    
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
                 scale_rewards=False, 
                 warm_up_steps=30, 
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
        self.beta = beta
        self.lmbda = lmbda
        self.eps = eps
        self.gpu_id = gpu_id
        self.accum_grad = accum_grad
        self.rlhf = True
        if reward_model is None:
            self.rlhf = False
        self.dist = params['env_params'].get("args").out_dist
        self.train_phase = params['train_phase']
        self.t = 0
        self.scale_rewards = scale_rewards
        self.warm_up = warm_up_steps
        self.init_model = None
        if init_model is not None:
            self.init_model = init_model.eval()
        self.prev_log_probs = None
        #self.prev_log_probs = {'noisy':None, 'clean':None}
        self.val_coef = val_coef
        self.en_coef = en_coef
        self.episode_len = run_steps
        self._r_mean = 0
        self._r2_mean = 0
        print(f"RLHF:{self.rlhf}")


    def run_episode(self, actor, critic, optimizer, n_epochs=3):
        policy = self.unroll_policy(actor, critic)
        return self.train_on_policy(policy, 
                                    actor, 
                                    critic, 
                                    optimizer,
                                    n_epochs)
        #return self.run_n_step_episode(batch, actor, critic, optimizer, n_epochs)

    
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
    

    def unroll_policy(self, actor, critic):
        #Set models to eval
        actor = actor.eval()
        #actor.set_evaluation(True)
        actor.set_evaluation(False)
        critic = critic.eval()

        rewards = []
        r_ts = []
        states = []
        logprobs = []
        actions = []
        cleans = []
        ep_kl_penalty = 0
        pretrain_loss = 0
      
        with torch.no_grad():
            for _ in range(self.episode_len):
                
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
                bs, ch, t, f = clean.shape

                action, log_probs, _, _ = actor.get_action(noisy)
                log_probs_1, _ = actor.get_action_prob(noisy, action)

                print(f"log_probs:{log_probs[0].mean(), log_probs[1].mean()}")
                print(f"log_probs_1:{log_probs_1[0].mean(), log_probs_1[1].mean()}")

                if self.init_model is not None:
                    init_action, _, _, _ = self.init_model.get_action(noisy)
                    ref_log_probs, _ = self.init_model.get_action_prob(noisy, action)
                    exp_state = self.env.get_next_state(state=noisy, action=init_action)
        
                state = self.env.get_next_state(state=noisy, action=action)
                state['cl_audio'] = cl_aud
                state['clean'] = clean
                if self.init_model is not None:
                    state['exp_est_audio'] = exp_state['est_audio']

                #Calculate kl_penalty
                ref_log_prob = None
                if self.init_model is not None:
                    ref_log_prob = ref_log_probs[0] + ref_log_probs[1][:, 0, :, :].permute(0, 2, 1) + ref_log_probs[1][:, 1, :, :].permute(0, 2, 1)
                log_prob = log_probs[0] + log_probs[1][:, 0, :, :].permute(0, 2, 1) + log_probs[1][:, 1, :, :].permute(0, 2, 1)
                print(f"log_prob:{log_prob.mean()}")

                if ref_log_prob is not None:
                    kl_penalty = torch.mean(log_prob - ref_log_prob, dim=[1, 2]).detach()
                    ratio = torch.exp(kl_penalty)
                    kl_penalty = ((ratio - 1) - kl_penalty).detach()
                    ep_kl_penalty += kl_penalty.mean()
                else:
                    kl_penalty = None
                
                #Store reward
                if self.rlhf:
                    r_t = self.env.get_RLHF_reward(state=state['noisy'].permute(0, 1, 3, 2), 
                                                   scale=self.scale_rewards)
                else:
                    r_t = self.env.get_PESQ_reward(state) 
                
                print(f"R:{r_t.reshape(-1)} | KL: {kl_penalty}")
                r_ts.append(r_t)

                #Supervised loss
                enhanced = state['noisy']
                enhanced_mag = torch.sqrt(enhanced[:, 0, :, :]**2 + enhanced[:, 1, :, :]**2)
                clean_mag = torch.sqrt(clean[:, 0, :, :]**2 + clean[:, 1, :, :]**2)
                
                mag_loss = (clean_mag - enhanced_mag)**2
                ri_loss = (clean - enhanced) ** 2
                supervised_loss = torch.mean(0.3 * ri_loss + 0.7 * mag_loss, dim=[1, 2, 3])

                pretrain_loss += supervised_loss.mean()
                
                print(f"r_t:{r_t.shape} kl:{kl_penalty.shape} loss:{supervised_loss.shape}")
                r_t = r_t - self.beta * kl_penalty - self.lmbda * supervised_loss
                
                #Store trajectory
                states.append(noisy)
                cleans.append(clean)
                rewards.append(r_t)

                actions.append(action)
                logprobs.append(log_prob)
                #logprobs.append(ref_log_prob)

            #Convert collected rewards to target_values and advantages
            rewards = torch.stack(rewards).reshape(bs, -1)
            r_ts = torch.stack(r_ts).reshape(-1)
            print(f"Rewards:{rewards}")
            target_values = self.get_expected_return(rewards)
            print(f"Returns:{target_values}")
            b_target = target_values.reshape(-1)
            advantages = self.get_advantages(target_values, states, critic)
            b_advantages = advantages.reshape(-1)
            
            states = torch.stack(states).reshape(-1, ch, t, f)
            cleans = torch.stack(cleans).reshape(-1, ch, t, f)
            
            actions = (([a[0][0] for a in actions], 
                        [a[0][1] for a in actions]), 
                        [a[1] for a in actions])
            
            actions = ((torch.stack(actions[0][0]).reshape(-1, f, t).detach(), 
                        torch.stack(actions[0][1]).reshape(-1, f, t).detach()),
                        torch.stack(actions[1]).reshape(-1, ch, t, f).detach())
            
            logprobs = torch.stack(logprobs).reshape(-1, f, t).detach()
            
            ep_kl_penalty = ep_kl_penalty / self.episode_len
            pretrain_loss = pretrain_loss / self.episode_len

        print(f"STATES        :{states.shape}")
        print(f"CLEAN         :{cleans.shape}")
        print(f"TARGET_VALS   :{b_target.shape}")
        print(f"ACTIONS       :{actions[0][0].shape, actions[0][1].shape, actions[1].shape}")
        print(f"LOGPROBS      :{logprobs.shape}")
        print(f"ADVANTAGES    :{b_advantages.shape}")
        print(f"POLICY RETURNS:{target_values.mean(0)}")

        print(f"STATES:{states.mean()} ACTION:{actions[0][0].mean(), actions[0][1].mean(), actions[1].mean()}")

        policy_out = {
            'states':states,
            'pretrain_loss':pretrain_loss, 
            'b_targets':b_target,
            'actions':actions,
            'log_probs':logprobs,
            'b_advantages':(b_advantages, advantages),
            'target_values':target_values,
            'r_ts':(r_ts, rewards),
            'ep_kl':ep_kl_penalty
        }
        
        return policy_out

    def train_on_policy(self, policy, actor, critic, optimizers, n_epochs):

        states = policy['states']
        pretrain_loss = policy['pretrain_loss']
        b_target = policy['b_targets']
        actions = policy['actions']
        logprobs = policy['log_probs']
        b_advantages, advantages = policy['b_advantages']
        target_values = policy['target_values']
        ep_kl_penalty = policy['ep_kl']
        r_ts, reward = policy['r_ts']
        
        #Start training over the unrolled batch of trajectories
        #Set models to train
        #NOTE: We don't want to set actor to train mode due to presence of layer/instance norm layers
        #acting differently in train and eval mode. PPO seems to be stable only when actor
        #is still in eval mode
        critic = critic.train()
        actor = actor.eval()
        actor.set_evaluation(False)

        a_optim, c_optim = optimizers
        
        step_clip_loss = 0
        step_val_loss = 0
        step_entropy_loss = 0
        step_pg_loss = 0
        VALUES = torch.zeros(target_values.shape)

        for _ in range(n_epochs):
            indices = [t for t in range(states.shape[0])]
            np.random.shuffle(indices)
            for t in range(0, len(indices), self.bs):
            
                #Get mini batch indices
                mb_indx = indices[t:t + self.bs]
                mb_states = states[mb_indx, ...]

                #Get new logprobs and values for the sampled (state, action) pair
                mb_action = ((actions[0][0][mb_indx, ...], actions[0][1][mb_indx, ...]), actions[1][mb_indx, ...])

                print(f"mb_STATES:{mb_states.mean()} mb_ACTION:{mb_action[0][0].mean(), mb_action[0][1].mean(), mb_action[1].mean()}")
                
                log_probs, entropies = actor.get_action_prob(mb_states, mb_action)
        
                values = critic(mb_states).reshape(-1)
                for i, val in enumerate(values):
                    b = mb_indx[i] // self.episode_len
                    ts = mb_indx[i] % self.episode_len
                    VALUES[b, ts] = val

                if self.train_phase:
                    entropy = entropies[0].permute(0, 2, 1) + entropies[1][:, 0, :, :] + entropies[1][:, 1, :, :]
                    log_prob = log_probs[0].permute(0, 2, 1) + log_probs[1][:, 0, :, :] + log_probs[1][:, 1, :, :]
                    old_log_prob = logprobs[mb_indx, ...].permute(0, 2, 1)
                else:
                    #ignore complex mask, just tune mag mask 
                    raise NotImplementedError
                
                print(f"log_prob:{log_prob.mean(), log_prob.shape}")
                print(f"old_logprob:{old_log_prob.mean(), old_log_prob.shape}")

                logratio = torch.mean(log_prob - old_log_prob, dim=[1, 2])
                ratio = torch.exp(logratio)
                print(f"Ratio:{ratio}")

                #Normalize advantages across minibatch
                mb_adv = b_advantages[mb_indx, ...]
                print(f"ADV:{mb_adv.mean()}")
                if self.bs > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-08)

                #Policy gradient loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                if pg_loss1.mean() == pg_loss2.mean():
                    pg_loss = pg_loss1.mean()
                else:
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                #value_loss
                v_loss = 0.5 * ((b_target[mb_indx] - values) ** 2).mean()

                #Entropy loss
                entropy_loss = entropy.mean()

                """
                #Supervised loss
                mb_act, _, _, _ = actor.get_action(mb_states)
                mb_next_state = self.env.get_next_state(state=mb_states, action=mb_act)
                
                mb_enhanced = mb_next_state['noisy']
                mb_enhanced_mag = torch.sqrt(mb_enhanced[:, 0, :, :]**2 + mb_enhanced[:, 1, :, :]**2)
               
                mb_clean = clean[mb_indx, ...] 
                mb_clean_mag = torch.sqrt(mb_clean[:, 0, :, :]**2 + mb_clean[:, 1, :, :]**2)

                supervised_loss = ((mb_clean - mb_enhanced) ** 2).mean() + ((mb_clean_mag - mb_enhanced_mag)**2).mean()
                """

                clip_loss = pg_loss #+ self.lmbda * supervised_loss - (self.en_coef * entropy_loss)
                
                print(f"clip_loss:{clip_loss.item()} pg_loss:{pg_loss}")
                wandb.log({
                    'ratio':ratio.mean(),
                    'pg_loss1':pg_loss1.mean(),
                    'pg_loss2':pg_loss2.mean(),
                })

                #optimizer.zero_grad()
                a_optim.zero_grad()
                c_optim.zero_grad()
                clip_loss.backward()
                v_loss.backward()
                #Update network
                if not (torch.isnan(clip_loss).any() or torch.isinf(clip_loss).any()) and (self.t % self.accum_grad == 0):
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    a_optim.step()
                    c_optim.step()

                step_clip_loss += clip_loss.item()
                step_pg_loss += pg_loss.item()
                step_val_loss += v_loss.item() 
                step_entropy_loss += entropy_loss.item()      
                self.t += 1
        
        print(f"Values:{VALUES.mean(0)}")

        step_clip_loss = step_clip_loss / (n_epochs * self.episode_len)
        step_pg_loss = step_pg_loss / (n_epochs * self.episode_len)
        step_val_loss = step_val_loss / (n_epochs * self.episode_len)                
        step_entropy_loss = step_entropy_loss / (n_epochs * self.episode_len)
        
                    
        return (step_clip_loss, step_val_loss, step_entropy_loss, step_pg_loss, pretrain_loss), \
               (target_values.mean(), VALUES.mean(), ep_kl_penalty, r_ts.mean(), reward.mean()), \
               advantages.mean()  

    '''
    def run_n_step_episode(self, batch, actor, critic, optimizer, n_epochs=3):
        """
        Imagine the episode N --> e1 --> e2 --> ... --> en --> Terminate
        Here the noisy signal is enhanced n times in an episode. 
        """
        #Preprocessed batch
        cl_aud, clean, noisy, _ = batch
        noisy = noisy.permute(0, 1, 3, 2)
        clean = clean.permute(0, 1, 3, 2)
        bs, ch, t, f = clean.shape
        ep_kl_penalty = 0

        #Set models to eval
        actor = actor.eval()
        critic = critic.eval()

        a_optim, c_optim = optimizer
        
        #Calculate target values and advantages
        curr = noisy
        rewards = []
        r_ts = []
        states = []
        logprobs = []
        actions = []
        
        with torch.no_grad():
            for _ in range(self.episode_len):
                #Unroll policy for n steps and store rewards.
                action, log_probs, _, _ = actor.get_action(curr)
                
                if self.init_model is not None:
                    init_action, _, _, _ = self.init_model.get_action(curr)
                    ref_log_probs, _ = self.init_model.get_action_prob(curr, action)
                    exp_state = self.env.get_next_state(state=curr, action=init_action)
        
                state = self.env.get_next_state(state=curr, action=action)
                state['cl_audio'] = cl_aud
                if self.init_model is not None:
                    state['exp_est_audio'] = exp_state['est_audio']

                #Calculate kl_penalty
                ref_log_prob = None
                if self.init_model is not None:
                    ref_log_prob = ref_log_probs[0] + ref_log_probs[1][:, 0, :, :].permute(0, 2, 1) + ref_log_probs[1][:, 1, :, :].permute(0, 2, 1)
                log_prob = log_probs[0] + log_probs[1][:, 0, :, :].permute(0, 2, 1) + log_probs[1][:, 1, :, :].permute(0, 2, 1)
                print(f"log_prob:{log_prob.shape}")

                if ref_log_prob is not None:
                    kl_penalty = torch.mean(log_prob - ref_log_prob, dim=[1, 2]).detach()
                    ratio = torch.exp(kl_penalty)
                
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    #old_approx_kl = (-kl_penalty).mean()
                    kl_penalty = ((ratio - 1) - kl_penalty).mean().detach()
                    ep_kl_penalty += kl_penalty
                else:
                    kl_penalty = None

                #Store reward
                if self.rlhf:
                    r_t = self.env.get_RLHF_reward(state=state['noisy'].permute(0, 1, 3, 2))  
                else:
                    r_t = self.env.get_PESQ_reward(state)

                print(f"R:{r_t.reshape(-1)} | KL: {kl_penalty}")
                r_ts.append(r_t)

                if self.beta > 0:
                    r_t = torch.max(r_t - self.beta * kl_penalty, 0)
                
                #Store trajectory
                states.append(curr)
                rewards.append(r_t)

                actions.append(action)
                logprobs.append(log_prob)
                #logprobs.append(ref_log_prob)
                curr = state['noisy']

            #Store the last enhanced state
            print(f"curr:{curr.shape}")
            states.append(curr)

            #Convert collected rewards to target_values and advantages
            rewards = torch.stack(rewards).reshape(bs, -1)
            r_ts = torch.stack(r_ts).reshape(-1)
            
            target_values = self.get_expected_return(rewards)
            b_target = target_values.reshape(-1)
            advantages = self.get_advantages(target_values, states, critic)
            b_advantages = advantages.reshape(-1)
            
            states = torch.stack(states)
            states = states[:-1, ...].reshape(-1, ch, t, f)
            clean = torch.stack([clean for _ in range(self.episode_len)]).reshape(-1, ch, t, f)
            
            actions = (([a[0][0] for a in actions], 
                        [a[0][1] for a in actions]), 
                        [a[1] for a in actions])
            
            actions = ((torch.stack(actions[0][0]).reshape(-1, f, t).detach(), 
                        torch.stack(actions[0][1]).reshape(-1, f, t).detach()),
                        torch.stack(actions[1]).reshape(-1, ch, t, f).detach())
            
            logprobs = torch.stack(logprobs).reshape(-1, f, t).detach()
            
            ep_kl_penalty = ep_kl_penalty / self.episode_len

        print(f"STATES        :{states.shape}")
        print(f"CLEAN         :{clean.shape}")
        print(f"TARGET_VALS   :{b_target.shape}")
        print(f"ACTIONS       :{actions[0][0].shape, actions[0][1].shape, actions[1].shape}")
        print(f"LOGPROBS      :{logprobs.shape}")
        print(f"ADVANTAGES    :{b_advantages.shape}")
        print(f"POLICY RETURNS:{target_values.mean(0)}")

        #Start training over the unrolled batch of trajectories
        #Set models to train
        #NOTE: We don't want to set actor to train mode due to presence of layer/instance norm layers
        #acting differently in train and eval mode. PPO seems to be stable only when actor
        #is still in eval mode
        if self.warm_up > 0:
            actor = actor.train()
        critic = critic.train()
        
        step_clip_loss = 0
        step_val_loss = 0
        step_entropy_loss = 0
        step_pg_loss = 0
        step_sup_loss = 0
        VALUES = torch.zeros(target_values.shape)

        for _ in range(n_epochs):
            indices = [t for t in range(states.shape[0])]
            np.random.shuffle(indices)
            for t in range(0, len(indices), bs):
            
                #Get mini batch indices
                mb_indx = indices[t:t+bs]
                mb_states = states[mb_indx, ...]

                #Get new logprobs and values for the sampled (state, action) pair
                mb_action = ((actions[0][0][mb_indx, ...], actions[0][1][mb_indx, ...]), actions[1][mb_indx, ...])
                
                log_probs, entropies = actor.get_action_prob(mb_states, mb_action)
        
                values = critic(mb_states).reshape(-1)
                for i, val in enumerate(values):
                    b = mb_indx[i] // self.episode_len
                    ts = mb_indx[i] % self.episode_len
                    VALUES[b, ts] = val

                if self.train_phase:
                    entropy = entropies[0].permute(0, 2, 1) + entropies[1][:, 0, :, :] + entropies[1][:, 1, :, :]
                    log_prob = log_probs[0].permute(0, 2, 1) + log_probs[1][:, 0, :, :] + log_probs[1][:, 1, :, :]
                    old_log_prob = logprobs[mb_indx, ...].permute(0, 2, 1)
                else:
                    #ignore complex mask, just tune mag mask 
                    raise NotImplementedError
                
                print(f"log_prob:{log_prob.mean(), log_prob.shape}")
                print(f"old_logprob:{old_log_prob.mean(), old_log_prob.shape}")

                logratio = torch.mean(log_prob - old_log_prob, dim=[1, 2])
                ratio = torch.exp(logratio)
                print(f"Ratio:{ratio}")

                #Normalize advantages across minibatch
                mb_adv = b_advantages[mb_indx]
                if bs > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-08)

                #Policy gradient loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                if pg_loss1.mean() == pg_loss2.mean():
                    pg_loss = pg_loss1.mean()
                else:
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                #value_loss
                v_loss = 0.5 * ((b_target[mb_indx] - values) ** 2).mean()

                #Entropy loss
                entropy_loss = entropy.mean()

                #Supervised loss
                mb_act, _, _, _ = actor.get_action(mb_states)
                mb_next_state = self.env.get_next_state(state=mb_states, action=mb_act)
                
                mb_enhanced = mb_next_state['noisy']
                mb_enhanced_mag = torch.sqrt(mb_enhanced[:, 0, :, :]**2 + mb_enhanced[:, 1, :, :]**2)
               
                mb_clean = clean[mb_indx, ...] 
                mb_clean_mag = torch.sqrt(mb_clean[:, 0, :, :]**2 + mb_clean[:, 1, :, :]**2)

                supervised_loss = ((mb_clean - mb_enhanced) ** 2).mean() + ((mb_clean_mag - mb_enhanced_mag)**2).mean()

                if self.warm_up > 0:
                    pg_loss = torch.tensor(0.0).to(self.gpu_id)
                    self.warm_up -= 1

                clip_loss = pg_loss + self.lmbda * supervised_loss #- (self.en_coef * entropy_loss)
                
                print(f"clip_loss:{clip_loss.item()} pg_loss:{pg_loss}")
                wandb.log({
                    'ratio':ratio.mean(),
                    'pg_loss1':pg_loss1.mean(),
                    'pg_loss2':pg_loss2.mean(),
                })

                #optimizer.zero_grad()
                a_optim.zero_grad()
                c_optim.zero_grad()
                clip_loss.backward()
                v_loss.backward()
                #Update network
                if not (torch.isnan(clip_loss).any() or torch.isinf(clip_loss).any()) and (self.t % self.accum_grad == 0):
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    a_optim.step()
                    c_optim.step()

                step_clip_loss += clip_loss.item()
                step_pg_loss += pg_loss.item()
                step_sup_loss += supervised_loss.item()
                step_val_loss += v_loss.item() 
                step_entropy_loss += entropy_loss.item()      
                self.t += 1
        
        print(f"Values:{VALUES.mean(0)}")

        step_clip_loss = step_clip_loss / (n_epochs * self.episode_len)
        step_pg_loss = step_pg_loss / (n_epochs * self.episode_len)
        step_val_loss = step_val_loss / (n_epochs * self.episode_len)                
        step_sup_loss = step_sup_loss / (n_epochs * self.episode_len)
        step_entropy_loss = step_entropy_loss / (n_epochs * self.episode_len)
        
                    
        return (step_clip_loss, step_val_loss, step_entropy_loss, step_pg_loss, step_sup_loss), \
               (target_values.sum(-1).mean(), VALUES.sum(-1).mean(), ep_kl_penalty, r_ts.sum(-1).mean()), \
               advantages.sum(-1).mean()  
    '''