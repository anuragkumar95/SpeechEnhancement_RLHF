# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import soundfile as sf

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True,
                        help="Root directory containing clean wav files.")
    parser.add_argument("--noise_dir", type=str, required=True,
                        help="Root directory containing noise wav files.")
    parser.add_argument("-o", "--output", type=str, required=False,
                        help="Directory to save dataset")
    parser.add_argument("--nisqa_pt", type=str, required=False,
                        help="Path to NISQA checkpoint.")
    
"""
python run_predict.py --mode predict_file --pretrained_model weights/nisqa.tar --deg /path/to/wav/file.wav --output_dir /path/to/dir/with/results
"""
class RANKING:
    """
    This class generates a ranking dataset for reward model training.
    """
    def __init__(self, clean_dir, noise_dir, nisqa_pt, out_dir, snr_low=-10, snr_high=10, K=5):
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        self.save_dir = out_dir
        self.nisqa_pt = nisqa_pt
        self.K = K
        self.snr = torch.distributions.Uniform(snr_low, snr_high)

    def mix_audios(self, clean, noise):
        snr = self.snr.sample()

        #calculate the amount of noise to add to get a specific snr
        alpha = (noise * torch.exp(snr / 20) - clean) / noise
        signal = clean + alpha * noise
        
        return signal
    
    def get_nisqa_score()
    
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
 
            for _ in range(self.K):
                #sample noise
                idx = np.random.choice(n_noise_samples)
                noise_file = noise_files[idx]
                noise_file = os.path.join(self.noise_dir, noise_files[idx])
                noise, n_sr = torchaudio.load(noise_file)

                assert n_sr == c_sr

                #mix them
                signal = self.mix_audios(clean, noise)

                #save
                sf.write(os.path.join(self.save_dir, clean_files[cidx]), signal, 16000)



if __name__ == "__main__":
    ARGS = args().parse_args()

    ranks = RANKING(clean_dir=ARGS.clean_dir, 
                    noise_dir=ARGS.noise_dir, 
                    nisqa_pt=ARGS.nisqa_pt, 
                    out_dir=ARGS.output)
    
    ranks.generate_rankings(n_size=15000)
