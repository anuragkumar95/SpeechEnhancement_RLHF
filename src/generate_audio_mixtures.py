# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import os
import ray
import logging
import pickle
import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
from torch.distributions import Normal
import torchaudio.functional as F
from tqdm import tqdm
import soundfile as sf
import itertools
from model.CMGAN.actor import TSCNet
from model.reward_model import RewardModel
from evaluation import run_enhancement_step, compute_metrics
from speech_enh_env import SpeechEnhancementAgent
from utils import preprocess_batch
#from ray.experimental import tqdm_ray

torch.manual_seed(123)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=False,
                        help="Root directory containing clean wav files.")
    parser.add_argument("--noise_dir", type=str, required=False,
                        help="Root directory containing noise wav files.")
    parser.add_argument("--mixture_dir", type=str, required=False,
                        help="Root directory containing mixture wav files.")
    parser.add_argument("--mos_file", type=str, required=False,
                        help="Path to file containing mos scores.")
    parser.add_argument("-o", "--output", type=str, required=False,
                        help="Directory to save dataset")
    parser.add_argument("-pt", "--model_pt", type=str, required=False,
                        help="Path to the CMGAN checkpoint.")
    parser.add_argument("-rpt", "--reward_pt", type=str, required=False, default=None,
                        help="Path to the CMGAN checkpoint.")
    parser.add_argument("-n", "--n_size", type=int, required=False, default=10000,
                        help="Number of mixtures to be produced.")
    parser.add_argument("-std", "--noise_std", type=float, required=False, default=0.01,
                        help="Variance of noise to be added.")
    parser.add_argument("-k", "--k", type=int, required=False, default=10, 
                        help="Number of mixtures per sample to be produced.")
    parser.add_argument("--mix_aud", action='store_true', required=False,
                        help="Set this flag to mix audios")
    parser.add_argument("--calc_pesq", action='store_true', required=False,
                        help="Set this flag to calculate pesq for generated mixtures.")
    parser.add_argument("--generate_ranks", action='store_true', required=False,
                        help="Set this flag to generate_ranks")
    parser.add_argument("--set", required=False, default='train',
                        help="Set this flag to generate_ranks for train set")
    parser.add_argument("--pre", action='store_true', 
                        help="Set this flag to load pretrained CMGAN checkpoint")
    return parser
    
torch.manual_seed(123)

class MixturesDataset:
    """
    This class generates a dataset for reward model training.
    """
    def __init__(self, clean_dir, noisy_dir, model_pt, out_dir, reward_pt=None, K=5, cutlen=40000, gpu_id=None, pre=True):
        #self.clean_dir = clean_dir
        #self.noisy_dir = noisy_dir
        self.clean_files = sorted([os.path.join(clean_dir, file) for file in os.listdir(clean_dir)])
        self.noisy_files = sorted([os.path.join(noisy_dir, file) for file in os.listdir(clean_dir)])
        self.save_dir = out_dir
        self.K = K
        #self.snr_means = [-15, -10, 5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
        self.model = TSCNet(num_channel=64, 
                            num_features=400 // 2 + 1,
                            distribution=None, 
                            gpu_id=gpu_id)
        
        checkpoint = torch.load(model_pt, map_location=torch.device('cpu'))
        if not pre:
            self.model.load_state_dict(checkpoint['actor_state_dict'])
        else:
            try:
                self.model.load_state_dict(checkpoint['generator_state_dict'])
            except KeyError as e:
                self.model.load_state_dict(checkpoint)
        

        self.model = self.model.to(gpu_id)
        self.model = self.model.eval()
        if not pre:
            self.model.set_evaluation(True)

        self.reward_model = None
        if reward_pt is not None:
            self.reward_model = RewardModel(in_channels=2)
            checkpoint = torch.load(reward_pt, map_location=torch.device('cpu'))
            self.reward_model.load_state_dict(checkpoint)
            self.reward_model = self.reward_model.to(gpu_id)
            self.reward_model = self.reward_model.eval()
       
        self.env = SpeechEnhancementAgent(n_fft=400,
                                          hop=100,
                                          gpu_id=gpu_id,
                                          args=None,
                                          reward_model=self.reward_model)
        self.cutlen = cutlen

    def mix_audios(self, clean, noise, snr):
        
        while noise.shape[-1] < clean.shape[-1]:
            noise = torch.cat([noise, noise], dim=-1)

        noise = noise[:, :clean.shape[-1]]

        #calculate the amount of noise to add to get a specific snr
        p_clean = (clean ** 2).mean()
        p_noise = (noise ** 2).mean()

        p_ratio = p_clean / p_noise
        alpha = torch.sqrt(p_ratio / (10 ** (snr / 10)))
        signal = clean + (alpha * noise)
        
        return signal.reshape(-1).cpu().numpy()

    def generate_k_samples(self, clean_file, noisy_file, save_metrics=True, noise_std=0.01):

        clean_wav, c_sr = torchaudio.load(clean_file)
        noisy_wav, n_sr = torchaudio.load(noisy_file)

        length = clean_wav.shape[-1]
        noisy_wav = noisy_wav[:, :length]

        if clean_wav.shape[-1] > self.cutlen:
            clean_wav = clean_wav[:, :self.cutlen]
            noisy_wav = noisy_wav[:, :self.cutlen]
            length = self.cutlen

        assert c_sr == n_sr == 16000

        batch = (clean_wav, noisy_wav, length)
        batch = preprocess_batch(batch, gpu_id=0, return_c=True)

        file_id = Path(noisy_file).stem

        for i in range(self.K):
            if i == 0:
                add_noise = False
            else:
                add_noise = True
            metrics = run_enhancement_step(self.env, 
                                           batch, 
                                           self.model, 
                                           length, 
                                           f"{file_id}_{i}.wav", 
                                           self.save_dir,
                                           save_metrics=save_metrics,
                                           save_track=True,
                                           add_noise=add_noise,
                                           noise_std=noise_std)
            if save_metrics:
                metrics_dir = os.path.join(self.save_dir, 'metrics')
                os.makedirs(metrics_dir, exist_ok=True)
                with open(os.path.join(metrics_dir, f"{file_id}_{i}.pickle"), 'wb') as f:
                    pickle.dump(metrics, f)
    
    def generate_mixtures(self, n_size=5000, std=0.01):
        n_clean_examples = len(self.clean_files)

        #sample clean indexes
        np.random.seed(123)
        cidxs = np.random.choice(n_clean_examples, n_size, replace=False)
        
        for i in tqdm(cidxs):
            self.generate_k_samples(self.clean_files[i], self.noisy_files[i], save_metrics=True, noise_std=std)

def calc_mixture_pesq(enhance_dir, clean_dir, save_dir):

    enhanced_files = os.listdir(enhance_dir)
    roots = ["_".join(file.split('_')[:2]) for file in enhanced_files]
    roots = list(set(roots))

    rand_roots = np.random.choice(roots, 50, replace=False)
    
    file_map = {}
    for root in rand_roots:
        if root not in file_map:
            file_map[root] = []
        for i in range(50):
            file_map[root].append(f"{root}_{i}")

    PESQ = {}
    for file_id in tqdm(file_map):
        clean_file = os.path.join(clean_dir, f"{file_id}.wav")
        clean_wav, _ = torchaudio.load(clean_file)
        for enh_file in file_map[file_id]:
            enh_file = os.path.join(enhance_dir, f"{enh_file}.wav")
            enh_wav, _ = torchaudio.load(enh_file)

            metrics = compute_metrics(clean_wav.reshape(-1).cpu().numpy(), 
                                        enh_wav.reshape(-1).cpu().numpy(), 
                                        16000, 
                                        0)
            if file_id not in PESQ:
                PESQ[file_id] = []
            PESQ[file_id].append(metrics[0])
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'pesq.pickle'), 'wb') as f:
        pickle.dump(PESQ, f)
                
def generate_ranking(mos_file, n_size, save_dir, set='train'):
    mixture_ids = {}
    
    mos = {}
    with open(mos_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            file_name, mos_score, _, _, _, _, _ = line.split(',')
            mos[file_name] = float(mos_score)
    
    FILES = [(file_name, mos_score) for file_name, mos_score in mos.items()]
    FILES = sorted(FILES, key=lambda x:x[1], reverse=True)
    num_files = len(FILES)
    print(f"TOTAL FILES:{len(FILES)}")
    written = 0 
    #n_size = 100000
    with open(os.path.join(save_dir, f'{set}.pairs'), 'w') as f:  
        #for (p1, m1) in tqdm(FILES):
        for _  in tqdm(range(n_size)):
            diff = -9999
            p1, p2 = None, None
            m1, m2 = None, None
            while(diff < 0.25):
                k = np.random.choice(num_files, 2, replace=False)
                (p1, m1), (p2, m2) = FILES[k[0]], FILES[k[1]]
                diff = m1 - m2
                
            f.write(f"{p1} {p2} {m1} {m2} {diff}\n")
        print(f"SAVED PAIRS:{n_size}")
            
    

if __name__ == "__main__":

    ARGS = args().parse_args()

    os.makedirs(ARGS.output, exist_ok=True)

    if ARGS.mix_aud:
        ranks = MixturesDataset(clean_dir=ARGS.clean_dir, 
                                noisy_dir=ARGS.noise_dir,  
                                out_dir=ARGS.output,
                                K=ARGS.k,
                                model_pt=ARGS.model_pt,
                                reward_pt=ARGS.reward_pt,
                                gpu_id=0,
                                pre=ARGS.pre)
        
        ranks.generate_mixtures(n_size=ARGS.n_size, std=ARGS.noise_std)

    if ARGS.calc_pesq:
        calc_mixture_pesq(enhance_dir=ARGS.mixture_dir, 
                          clean_dir=ARGS.clean_dir, 
                          save_dir=ARGS.output)

    if ARGS.generate_ranks:
        generate_ranking(mos_file=ARGS.mos_file, 
                         n_size=ARGS.n_size,
                         save_dir=ARGS.output,
                         set=ARGS.set)

