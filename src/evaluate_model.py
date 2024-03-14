from model.actor import TSCNet
from model.reward_model import RewardModel
from model.critic import QNet
#from model.cmgan import TSCNet
from RLHF import REINFORCE, PPO

import copy
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

import torch
import os

import pickle
from speech_enh_env import SpeechEnhancementAgent

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True,
                        help="Root directory to Voicebank.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for results. Will create one if doesn't exist")
    parser.add_argument("-pt", "--ckpt", type=str, required=False, default=None,
                        help="Path to saved checkpoint to evaluate.")
    parser.add_argument("--pre", action="store_true")
    parser.add_argument("-rpt", "--reward_pt", type=str, required=False, default=None,
                        help="Path to saved rewardmodel checkpoint to evaluate.")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--save_actions", action='store_true', 
                        help="Flag to save actions")
    parser.add_argument("--save_specs", action='store_true', 
                        help="Flag to save enhanced spectograms")
    parser.add_argument("--save_scores", action='store_true', 
                        help="Flag to save critic scores")
    parser.add_argument("--save_pesq", action='store_true', 
                        help="Flag to save pesq values")
    parser.add_argument("--save_rewards", action='store_true',
                        help='Flag to save rewards from the reward model.')
    return parser

class EvalModel:
    def __init__(self, modes, save_path, pre, model_pt=None, reward_pt=None, gpu_id=None):
        self.modes = modes
        self.n_fft = 400
        self.hop = 100
        
        self.actor = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1,
                            distribution="Normal", 
                            gpu_id=gpu_id)
        
        self.critic = None
        self.reward_model = None

        if model_pt is not None:
            self.critic = QNet(ndf=16, in_channel=2, out_channel=1)
            checkpoint = torch.load(model_pt, map_location=torch.device('cpu'))
            if pre:
                self.actor.load_state_dict(checkpoint['generator_state_dict'])
            else:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f"Loaded checkpoint from {model_pt}...")

            if gpu_id is not None:
                self.actor = self.actor.to(gpu_id)
                self.critic = self.critic.to(gpu_id)

            self.actor.eval()
            self.critic.eval()

        if reward_pt is not None:
            self.reward_model = RewardModel(policy=copy.deepcopy(self.actor))
            reward_checkpoint = torch.load(reward_pt, map_location=torch.device('cpu'))
            self.reward_model.load_state_dict(reward_checkpoint)
            print(f"Loaded reward model from {reward_pt} ... ")
            
            if gpu_id is not None:
                self.reward_model = self.reward_model.to(gpu_id)

            self.reward_model.eval()

        self.env = SpeechEnhancementAgent(n_fft=self.n_fft,
                                          hop=self.hop,
                                          gpu_id=gpu_id,
                                          args=None,
                                          reward_model=self.reward_model)

        self.save_path = save_path
        self.gpu_id = gpu_id

    def evaluate(self, dataset):
        
        with torch.no_grad():
            for mode in self.modes:
                save_path = f"{self.save_path}/{mode}"
                os.makedirs(save_path, exist_ok=True)
                for i, batch in enumerate(dataset):
                    
                    _, noisy_aud, _ = batch
                    #Preprocess batch
                    batch = preprocess_batch(batch, gpu_id=self.gpu_id)

                    cl_aud, clean, noisy, _ = batch
                    inp = noisy.permute(0, 1, 3, 2)

                    #Forward pass through actor to get the action(mask)
                    action, _, _ = self.actor.get_action(inp)

                    #Apply action  to get the next state
                    next_state = self.env.get_next_state(state=inp, 
                                                        action=action)
                    if mode == 'action':
                        with open(os.path.join(save_path, f"action_{i}.pickle"), 'wb') as f:
                            action = (action[0].detach().cpu().numpy(), action[1].detach().cpu().numpy())
                            pickle.dump(action, f)
                        print(f"action_{i}.pickle saved in {save_path}")

                    if mode == 'spectogram':
                        with open(os.path.join(save_path, f"spec_{i}.pickle"), 'wb') as f:
                            spec = {
                                'enhanced': next_state['noisy'].detach().cpu().numpy(),
                                'noisy'   : noisy.detach().cpu().numpy(),
                                'clean'   : clean.detach().cpu().numpy()
                            }
                            pickle.dump(spec, f)
                        print(f"spec_{i}.pickle saved in {save_path}")

                    if mode == 'critic_score':
                        score_clean = self.critic(clean)
                        score_noisy = self.critic(noisy)
                        score_enhanced = self.critic(next_state['noisy'])
                        scores = {
                            'enhanced' : score_enhanced.detach().cpu().numpy(),
                            'noisy'    : score_noisy.detach().cpu().numpy(),
                            'clean'    : score_clean.detach().cpu().numpy()
                        }
                        with open(os.path.join(save_path, f"score_{i}.pickle"), 'wb') as f:
                            pickle.dump(scores, f)
                        print(f"score_{i}.pickle saved in {save_path}")

                    if mode == 'pesq':
                        enh_aud = next_state['est_audio'].detach().cpu().numpy()
                        
                        n_pesq, pesq_mask = batch_pesq(cl_aud.detach().cpu().numpy(), noisy_aud.detach().cpu().numpy())
                        n_pesq = (n_pesq * pesq_mask).mean()

                        e_pesq, pesq_mask = batch_pesq(cl_aud.detach().cpu().numpy(), enh_aud)
                        e_pesq = (e_pesq * pesq_mask).mean()

                        pesq = {
                            'noisy':original_pesq(n_pesq),
                            'enhanced':original_pesq(e_pesq),
                        }

                        with open(os.path.join(save_path, f"pesq_{i}.pickle"), 'wb') as f:
                            pickle.dump(pesq, f)
                        print(f"pesq_{i}.pickle saved in {save_path}")

                    if mode == 'rewards':
                        rewards = {1:{}, 2:{}}

                        noisy_reward_1 = self.reward_model.get_reward(inp=noisy, out=noisy, mode=1)
                        clean_reward_1 = self.reward_model.get_reward(inp=noisy, out=clean, mode=1)

                        noisy_reward_2 = self.reward_model.get_reward(inp=noisy, out=noisy, mode=2)
                        clean_reward_2 = self.reward_model.get_reward(inp=noisy, out=clean, mode=2)

                        rewards[1] = {
                            'noisy': noisy_reward_1,
                            'clean': clean_reward_1
                        }

                        rewards[2] = {
                            'noisy': noisy_reward_2,
                            'clean': clean_reward_2
                        }

                        with open(os.path.join(save_path, f"reward_{i}.pickle"), 'wb') as f:
                            pickle.dump(rewards, f)
                        print(f"reward_{i}.pickle saved in {save_path}")


if __name__ == '__main__':
    ARGS = args().parse_args()

    modes = []
    if ARGS.save_actions:
        modes.append('action')
    if ARGS.save_specs:
        modes.append('spectogram')
    if ARGS.save_scores:
        modes.append('critic_score')
    if ARGS.save_pesq:
        modes.append('pesq')
    if ARGS.save_rewards:
        modes.append('save_rewards')

    eval = EvalModel(modes=modes, 
                    model_pt=ARGS.ckpt, 
                    save_path=ARGS.output, 
                    pre=args.pre,
                    gpu_id=0)
    
    _, test_ds = load_data(ARGS.root, 
                           ARGS.batchsize, 
                           1, 
                           40000,
                           gpu = False,
                           shuffle=False)
    
    eval.evaluate(test_ds)

                    
