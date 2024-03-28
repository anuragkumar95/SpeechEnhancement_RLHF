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
import torchaudio.functional as F
from tqdm import tqdm
import soundfile as sf
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
    return parser
    

def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

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
        self.snr = torch.distributions.Uniform(snr_low, snr_high)
        ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)

    def mix_audios(self, clean, noise):
        
        if clean.shape[-1] != noise.shape[-1]:
            while noise.shape[-1] >= clean.shape[-1]:
                noise = torch.cat([noise, noise], dim=-1)
        noise = noise[:, :clean.shape[-1]]

        snr = self.snr.sample()
        snr = float("{:.2f}".format(snr))

        #calculate the amount of noise to add to get a specific snr
        p_clean = (clean ** 2).mean()
        p_noise = (noise ** 2).mean()

        p_ratio = p_clean / p_noise
        alpha = torch.sqrt(p_ratio * (10 ** (-snr / 10)))
        signal = clean + (alpha * noise)
        
        return signal.reshape(-1).cpu().numpy(), snr
    
    @ray.remote
    def generate_k_samples(self, n_clean_examples, n_noise_samples):
        #sample a clean audio
        cidx = np.random.choice(n_clean_examples)
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

            #assert n_sr == c_sr

            #mix them
            signal, snr = self.mix_audios(clean, noise)

            #save
            sf.write(os.path.join(self.save_dir, f"{self.clean_files[cidx][:-len('.wav')]}-{i}_snr_{snr}.wav"), signal, 16000)
        bar.update.remote(1)
    
    def generate_mixtures(self, n_size=5000):
        n_clean_examples = len(self.clean_files)
        n_noise_samples = len(self.noise_files)
        
        futures = [self.generate_k_samples.remote(n_clean_examples, n_noise_samples) for _ in range(n_size)]
        
        ret = []
        for fut in tqdm(to_iterator(futures), total=len(futures)):
            ret.append(fut)
        
        ray.get(futures)
        ray.shutdown()
        


def generate_ranking(mos_file, mixture_dir, save_dir):
    mixture_ids = {}
    for file in os.listdir(mixture_dir):
        _id_ = file.split('-')[0]
        if _id_ not in mixture_ids:
            mixture_ids[_id_] = []
        mixture_ids[_id_].append(file)

    mos = {}
    with open(mos_file, 'r') as f:
        lines = f.readines()
        for line in lines:
            file_name, mos_score, _, _, _, _ = line.split(',')
            mos[file_name] = float(mos_score)

    with open(os.path.join(save_dir, 'ranks'), 'w') as f:
        for _id_ in mixture_ids:
            ranks = [(i, mos[i]) for i in mixture_ids[_id_]]
            sorted_ranks = sorted(ranks, key=lambda x:x[1])
            sorted_file_ids = [i[0] for i in sorted_ranks]
            line = " ".join(sorted_file_ids)
            f.write(f"{line}\n")


if __name__ == "__main__":

    ARGS = args().parse_args()

    if ARGS.mix_aud:
        ranks = MixturesDataset(clean_dir=ARGS.clean_dir, 
                                noise_dir=ARGS.noise_dir,  
                                out_dir=ARGS.output)
        
        ranks.generate_mixtures(n_size=ARGS.n_size)

    if ARGS.generate_ranks:
        generate_ranking(mos_file=ARGS.mos_file, 
                         mixture_dir=ARGS.mixture_dir, 
                         save_dir=ARGS.output)

