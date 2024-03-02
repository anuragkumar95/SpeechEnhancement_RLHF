from model.actor import TSCNet, RewardModel
from model.critic import QNet
#from model.cmgan import TSCNet
from RLHF import REINFORCE, PPO

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
    parser.add_argument("-pt", "--ckpt", type=str, required=True, default=None,
                        help="Path to saved checkpoint to evaluate.")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--save_actions", action='store_true', 
                        help="Flag to save actions")
    parser.add_argument("--save_specs", action='store_true', 
                        help="Flag to save enhanced spectograms")
    parser.add_argument("--save_scores", action='store_true', 
                        help="Flag to save critic scores")
    return parser

class EvalModel:
    def __init__(self, modes, model_pt, save_path, gpu_id=None):
        self.modes = modes

        self.n_fft = 400
        self.hop = 100

        self.actor = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1,
                            distribution=True, 
                            gpu_id=gpu_id)
        
        self.critic = QNet(ndf=16, in_channel=2, out_channel=1)

        checkpoint = torch.load(model_pt, map_location=torch.device('cpu'))
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded checkpoint from {model_pt}...")

        if gpu_id is not None:
            self.actor = self.actor.to(gpu_id)
            self.critic = self.critic.to(gpu_id)

        self.actor.eval()
        self.critic.eval()

        self.env = SpeechEnhancementAgent(n_fft=self.n_fft,
                                          hop=self.hop,
                                          gpu_id=gpu_id,
                                          args=None,
                                          reward_model=None)

        self.save_path = save_path
        self.gpu_id = gpu_id

    def evaluate(self, dataset):
        
        with torch.no_grad():
            for mode in self.modes:
                save_path = f"{self.save_path}/{mode}"
                os.makedirs(save_path, exist_ok=True)
                for i, batch in enumerate(dataset):
                    
                    #Preprocess batch
                    batch = preprocess_batch(batch, gpu_id=self.gpu_id)

                    _, clean, noisy, _ = batch
                    inp = noisy.permute(0, 1, 3, 2)

                    #Forward pass through actor to get the action(mask)
                    action, _, _ = self.actor.get_action(inp)

                    if mode == 'action':
                        with open(os.path.join(save_path, f"action_{i}.pickle"), 'wb') as f:
                            action = (action[0].detach().cpu().numpy(), action[1].detach().cpu().numpy())
                            pickle.dump(f, action)
                        print(f"action_{i}.pickle saved in {save_path}")

                    #Apply action  to get the next state
                    next_state = self.env.get_next_state(state=inp, 
                                                        action=action)
                    if mode == 'spectogram':
                        with open(os.path.join(save_path, f"spec_{i}.pickle"), 'wb') as f:
                            spec = {
                                'enhanced': next_state['est_audio'].detach().cpu().numpy(),
                                'noisy'   : noisy.detach().cpu().numpy(),
                                'clean'   : clean.detach().cpu().numpy()
                            }
                            pickle.dump(f, spec)
                        print(f"spec_{i}.pickle saved in {save_path}")

                    if mode == 'critic_score':
                        score_clean = self.critic(clean)
                        score_noisy = self.critic(noisy)
                        score_enhanced = self.critic(next_state['est_audio'])
                        scores = {
                            'enhanced' : score_enhanced.detach().cpu().numpy(),
                            'noisy'    : score_noisy.detach().cpu().numpy(),
                            'clean'    : score_clean.detach().cpu().numpy()
                        }
                        with open(os.path.join(save_path, f"score_{i}.pickle"), 'wb') as f:
                            pickle.dump(f, scores)
                        print(f"score_{i}.pickle saved in {save_path}")
                        

if __name__ == '__main__':
    ARGS = args().parse_args()

    modes = []
    if ARGS.save_actions:
        modes.append('action')
    if ARGS.save_specs:
        modes.append('spectogram')
    if ARGS.save_scores:
        modes.append('critic_score')


    eval = EvalModel(modes=modes, 
                    model_pt=ARGS.ckpt, 
                    save_path=ARGS.output, 
                    gpu_id=0)
    
    _, test_ds = load_data(ARGS.root, 
                           ARGS.batchsize, 
                           1, 
                           40000,
                           gpu = False)
    
    eval.evaluate(test_ds)

                    
