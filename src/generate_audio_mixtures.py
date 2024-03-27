# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm
import soundfile as sf

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
    
class MixturesDataset:
    """
    This class generates a dataset for reward model training.
    """
    def __init__(self, clean_dir, noise_dir, nisqa_pt, out_dir, snr_low=-10, snr_high=10, K=5):
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        self.save_dir = out_dir
        self.nisqa_pt = nisqa_pt
        self.K = K
        self.snr = torch.distributions.Uniform(snr_low, snr_high)

    def mix_audios(self, clean, noise):
        
        if clean.shape[-1] != noise.shape[-1]:
            while noise.shape[-1] >= clean.shape[-1]:
                noise = torch.cat([noise, noise], dim=-1)
        noise = noise[:, :clean.shape[-1]]

        snr = self.snr.sample()

        #calculate the amount of noise to add to get a specific snr
        p_clean = (clean ** 2).mean()
        p_noise = (noise ** 2).mean()

        p_ratio = p_clean / p_noise
        alpha = torch.sqrt(p_ratio * (10 ** (-snr / 10)))
        signal = clean + (alpha * noise)
        
        return signal.cpu().numpy()
    
    def generate_mixtures(self, n_size=15000):

        clean_files = os.listdir(self.clean_dir)
        noise_files = os.listdir(self.noise_dir)
        n_clean_examples = len(clean_files)
        n_noise_samples = len(noise_files)

        for i in range(n_size):
            #sample a clean audio
            cidx = np.random.choice(n_clean_examples)
            clean_file = os.path.join(self.clean_dir, clean_files[cidx])
            clean, c_sr = torchaudio.load(clean_file)

            assert c_sr == 16000
 
            for i in range(self.K):
                #sample noise
                idx = np.random.choice(n_noise_samples)
                noise_file = noise_files[idx]
                noise_file = os.path.join(self.noise_dir, noise_files[idx])
                noise, n_sr = torchaudio.load(noise_file)

                if n_sr != c_sr:
                    noise = F.resample(noise, orig_freq=n_sr, new_freq=c_sr)

                assert n_sr == c_sr

                #mix them
                signal = self.mix_audios(clean, noise)

                #save
                sf.write(os.path.join(self.save_dir, f"{clean_files[cidx]}-{i}"), signal, 16000)


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

