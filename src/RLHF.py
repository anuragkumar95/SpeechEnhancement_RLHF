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
from torch.distributions import Normal

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
    Base class for PPO policy gradient method.
    """
    def __init__(self, 
                 init_model, 
                 reward_model, 
                 gpu_id=None, 
                 run_steps=1, 
                 beta=0.2,
                 eps=0.01, 
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
        self.eps = eps
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


    def run_episode(self, batch, actor, critic, optimizer):
        if self.episode_len <= 1:
            return self.run_one_step_episode(batch, actor, critic, optimizer)
        else:
            return self.run_n_step_episode(batch, actor, critic, optimizer)

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
    
    def get_action_prob(self, mu, logvar, action):
        sigma = torch.exp(0.5 * logvar) + 1e-08
        dist = Normal(mu, sigma)
        return dist.log_prob(action)

    def run_n_step_episode(self, batch, actor, critic, optimizer):
        """
        Imagine the episode N --> e1 --> e2 --> ... --> en --> Terminate
        Here the noisy signal is enhanced n times in an episode. 
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
            r_ts = []
            states = []
            logprobs = []
            actions = []
            for _ in range(self.episode_len):
                #Unroll policy for n steps and store rewards.
                action, log_probs, _, params = actor.get_action(curr)
                init_action, ref_log_probs, _, ref_params = self.init_model.get_action(curr)
                print(f"REF1:{ref_log_probs[0].mean(), ref_log_probs[1].mean()}")
                print(f"INIT:{init_action[0].shape, init_action[0].mean()} ACT:{action[0].shape, action[0].mean()}")
              
                ref_log_probs2, _ = self.init_model.get_action_prob(curr, init_action)
                log_probs2, _ = self.init_model.get_action_prob(curr, action)
                
                print(f"REF2:{ref_log_probs2[0].mean()}")
                print(f"LOG2:{log_probs2[0].mean()}")
                
                state = self.env.get_next_state(state=curr, action=action)
                exp_state = self.env.get_next_state(state=curr, action=init_action)
                state['cl_audio'] = cl_aud
                state['exp_est_audio'] = exp_state['est_audio']
                state['clean'] = clean

                #Calculate kl_penalty
                
                ref_log_prob = ref_log_probs[0] + ref_log_probs[1][:, 0, :, :].permute(0, 2, 1) + ref_log_probs[1][:, 1, :, :].permute(0, 2, 1)
                log_prob = log_probs[0] + log_probs[1][:, 0, :, :].permute(0, 2, 1) + log_probs[1][:, 1, :, :].permute(0, 2, 1)
                kl_penalty = torch.mean(log_prob, dim=[1, 2]) - torch.mean(ref_log_prob, dim=[1, 2])
                ep_kl_penalty += kl_penalty.detach()
                
                #Store reward
                if self.rlhf:
                    r_t = self.env.get_RLHF_reward(inp=curr, out=state['noisy'].permute(0, 1, 3, 2))    
                else:
                    r_t = self.env.get_PESQ_reward(state)
                print(f"R:{r_t.reshape(-1)} | KL:{kl_penalty.reshape(-1)}")
                
                #Store trajectory
                states.append(curr)
                rewards.append(r_t - self.beta * kl_penalty)
                for i in range(bs):
                    act = {
                        'action':(action[0][i, ...].detach(), action[1][i, ...].detach()),
                    }
                    actions.append(act)
                    logprobs.append((log_probs[0][i, ...].detach(), log_probs[1][i, ...].detach()))
                
                r_ts.append(r_t)

            #Convert collected rewards to target_values and advantages
            rewards = torch.stack(rewards).reshape(bs, -1)
            r_ts = torch.stack(r_ts).reshape(-1)
            target_values = self.get_expected_return(rewards)
            advantages = self.get_advantages(target_values, states, critic)
            ep_kl_penalty = ep_kl_penalty / self.episode_len
            
            #flatten all
            b_target_values = target_values.reshape(-1)
            b_advantages = advantages.reshape(-1)
            states = torch.stack(states)
            step, b, c, t, f = states.shape
            states = states.reshape(step * b, c, t, f)

        print(f"STATES        :{states.shape}")
        print(f"TARGET_VALS   :{b_target_values.shape}")
        print(f"ACTIONS       :{len(actions)}")
        print(f"LOGPROBS      :{len(logprobs)}")
        print(f"Policy returns:{target_values.mean(0)}")

        #Start training over the unrolled batch of trajectories
        actor.train()
        critic.train()
        step_clip_loss = 0
        step_val_loss = 0
        step_entropy_loss = 0
        step_pg_loss = 0
        VALUES = torch.zeros(target_values.shape)

        indices = [t for t in range(len(states))]
        np.random.shuffle(indices)

        for t in range(0, len(indices), bs):
            #Get mini batch indices
            mb_indx = indices[t:t+bs]
            mb_states = states[mb_indx, ...]

            #Get new logprobs and values for the sampled (state, action) pair
            mb_action = ([actions[i]['action'][0] for i in mb_indx],
                         [actions[i]['action'][1] for i in mb_indx])
            mb_action = (torch.stack(mb_action[0]).squeeze(1), torch.stack(mb_action[1]))

            log_probs, entropies = actor.get_action_prob(mb_states, mb_action)
            values = critic(mb_states).reshape(-1)
            for i, val in enumerate(values):
                b = mb_indx[i] // self.episode_len
                ts = mb_indx[i] % self.episode_len
                VALUES[b, ts] = val

            if self.train_phase:
                entropy = 0
                #entropy = entropies[0] + entropies[1][:, 0, :, :] + entropies[1][:, 1, :, :]
                log_prob = log_probs[0].permute(0, 2, 1) + log_probs[1][:, 0, :, :] + log_probs[1][:, 1, :, :]

                print(f"log_prob:{log_probs[0].mean(), log_probs[1].mean()}")


                old_logprobs = ([logprobs[i][0] for i in mb_indx],
                                [logprobs[i][1] for i in mb_indx])
                mb_oldlogprobs = (torch.stack(old_logprobs[0]), torch.stack(old_logprobs[1]))
                old_log_prob = mb_oldlogprobs[0].permute(0, 2, 1) + mb_oldlogprobs[1][:, 0, :, :] + mb_oldlogprobs[1][:, 1, :, :]
                
            else:
                #ignore complex mask, just tune mag mask 
                entropy = entropies[0]
                log_prob, old_log_prob = log_probs[0], mb_oldlogprobs[0].permute(0, 2, 1)
            
            print(f"old_logprob:{old_log_prob.mean()}")
            logratio = torch.mean(log_prob - old_log_prob, dim=[1, 2]) 
            ratio = torch.exp(logratio)
            
            #Policy loss
            pg_loss1 = -b_advantages[mb_indx] * ratio
            pg_loss2 = -b_advantages[mb_indx] * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            #value_loss
            v_loss = 0.5 * ((b_target_values[mb_indx] - values) ** 2).mean()

            #Entropy loss
            entropy_loss = entropy#.mean()

            clip_loss = pg_loss - (self.en_coef * entropy_loss) + (self.val_coef * v_loss)

            wandb.log({
                'ratio':ratio.mean(),
                'pg_loss1':pg_loss1.mean(),
                'pg_loss2':pg_loss2.mean()
            })

            optimizer.zero_grad()
            clip_loss.backward()
            #Update network
            if not (torch.isnan(clip_loss).any() or torch.isinf(clip_loss).any()) and (self.t % self.accum_grad == 0):
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()

            step_clip_loss += clip_loss.item()
            step_pg_loss += pg_loss.item()
            step_val_loss += v_loss.item()
            #step_entropy_loss += entropy_loss.item()        
            self.t += 1
        
        print(f"Values:{VALUES.mean(0)}")

        step_clip_loss = step_clip_loss / self.episode_len
        step_pg_loss = step_pg_loss / self.episode_len
        step_val_loss = step_val_loss / self.episode_len
        #step_entropy_loss = step_entropy_loss / self.episode_len
        
                    
        return (step_clip_loss, step_val_loss, step_entropy_loss, step_pg_loss), (target_values.sum(-1).mean(), VALUES.sum(-1).mean(), ep_kl_penalty.mean(), r_ts.sum(-1).mean()), advantages.sum(-1).mean()

            

                

            

        