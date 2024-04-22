# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import os
import ray
import logging
import argparse
import numpy as np
import torch
import torchaudio
from torch.distributions import Normal
import torchaudio.functional as F
from tqdm import tqdm
import soundfile as sf
import itertools
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
    parser.add_argument("-n", "--n_size", type=int, required=False, default=10000, 
                        help="Number of mixtures to be produced.")
    parser.add_argument("--mix_aud", action='store_true', required=False,
                        help="Set this flag to mix audios")
    parser.add_argument("--generate_ranks", action='store_true', required=False,
                        help="Set this flag to generate_ranks")
    parser.add_argument("--set", required=False, default='train',
                        help="Set this flag to generate_ranks for train set")
    return parser
    

class MixturesDataset:
    """
    This class generates a dataset for reward model training.
    """
    def __init__(self, clean_dir, noise_dir, out_dir, snr_low=-10, snr_high=10, K=5):
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        self.clean_files = os.listdir(clean_dir)
        self.noise_files = os.listdir(noise_dir)
        self.save_dir = out_dir
        self.K = K
        self.snr_means = [snr_low + ((snr_high - snr_low) * i/K) for i in range(K)]

    def mix_audios(self, clean, noise, snr):
        
        if clean.shape[-1] != noise.shape[-1]:
            while noise.shape[-1] >= clean.shape[-1]:
                noise = torch.cat([noise, noise], dim=-1)
        noise = noise[:, :clean.shape[-1]]

        #calculate the amount of noise to add to get a specific snr
        p_clean = (clean ** 2).mean()
        p_noise = (noise ** 2).mean()

        p_ratio = p_clean / p_noise
        alpha = torch.sqrt(p_ratio / (10 ** (snr / 20)))
        signal = clean + (alpha * noise)
        
        return signal.reshape(-1).cpu().numpy()
    
    def generate_k_samples(self, cidx, n_noise_samples):
    
        clean_file = os.path.join(self.clean_dir, self.clean_files[cidx])
        clean, c_sr = torchaudio.load(clean_file)

        assert c_sr == 16000

        for i in range(self.K):
            #sample noise
            idx = np.random.choice(n_noise_samples)
            noise_file = self.noise_files[idx]
            noise_file = os.path.join(self.noise_dir, self.noise_files[idx])
            noise, n_sr = torchaudio.load(noise_file)

            if n_sr != c_sr:
                noise = F.resample(noise, orig_freq=n_sr, new_freq=c_sr)

            #mix them
            snr_dist = Normal(self.snr_means[i], 3.0)
            snr = snr_dist.sample()
            signal = self.mix_audios(clean, noise, snr)
            
            #save
            snr = float("{:.2f}".format(snr))
            sf.write(os.path.join(self.save_dir, f"{self.clean_files[cidx][:-len('.wav')]}-{i}_{self.noise_files[idx][:-len('.wav')]}_snr_{snr}.wav"), signal, 16000)

    
    def generate_mixtures(self, n_size=5000):
        n_clean_examples = len(self.clean_files)
        n_noise_samples = len(self.noise_files)
        #sample clean indexes
        cidxs = np.random.choice(n_clean_examples, n_size, replace=False)
        
        for i in tqdm(range(n_size)):
            self.generate_k_samples(cidxs[i], n_noise_samples)


def generate_ranking(mos_file, mixture_dir, save_dir, set='train'):
    mixture_ids = {}
    for file in os.listdir(mixture_dir):
        file_id = file[:-len(".wav")]
        if "enh" not in file_id:
            _id_ = file_id.split('-')[0]
        else:
            _id_ = file_id[len("enh_"):]
        if _id_ not in mixture_ids:
            mixture_ids[_id_] = []
        mixture_ids[_id_].append(file)

    mos = {}
    with open(mos_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            file_name, mos_score, _, _, _, _, _ = line.split(',')
            mos[file_name] = float(mos_score)

    with open(os.path.join(save_dir, f'{set}.ranks'), 'w') as f:
        for _id_ in mixture_ids:
            if len(mixture_ids[_id_]) > 1:
                ranks = [(i, mos[i]) for i in mixture_ids[_id_]]
                sorted_ranks = sorted(ranks, key=lambda x:x[1], reverse=True)
                sorted_file_ids = [i[0] for i in sorted_ranks]
                line = " ".join(sorted_file_ids)
                f.write(f"{line}\n")
"""

def generate_ranking(mos_file, mixture_dir, save_dir, set='train'):
    mos = []
    with open(mos_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            file_name, mos_score, _, _, _, _, _ = line.split(',')
            #mos[file_name] = float(mos_score)
            mos.append((file_name, float(mos_score)))
            

    mos = sorted(mos, key=lambda x:x[1], reverse=True)

    if set == 'train':
        n = 45000
    if set == 'test':
        n = 1000
    with open(os.path.join(save_dir, f'{set}.ranks'), 'w') as f:
        for i in range(n):
            ranks = itertools.combinations(mos, 5)
            sorted_file_ids = [i[0] for i in ranks]
            line = " ".join(sorted_file_ids)
            f.write(f"{line}\n")
"""   

if __name__ == "__main__":

    ARGS = args().parse_args()

    if ARGS.mix_aud:
        ranks = MixturesDataset(clean_dir=ARGS.clean_dir, 
                                noise_dir=ARGS.noise_dir,  
                                out_dir=ARGS.output,
                                snr_low=-10, 
                                snr_high=40)
        
        ranks.generate_mixtures(n_size=ARGS.n_size)

    if ARGS.generate_ranks:
        generate_ranking(mos_file=ARGS.mos_file, 
                         mixture_dir=ARGS.mixture_dir, 
                         save_dir=ARGS.output,
                         set=ARGS.set)

