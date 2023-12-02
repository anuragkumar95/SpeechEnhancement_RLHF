# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
Created on 23rd Nov, 2023
"""

import torch

from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F
import os
import numpy as np
import tempfile
import gzip


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data.distributed import DistributedSampler

class JNDDataset(Dataset):
    def __init__(self, root, path_root, indices, cut_len=40000, resample=False, shift=False):
        self.data_root = root
        self.indices = indices
        self.resample = resample
        self.paths = self.collect_paths(path_root)
        self.cut_len = cut_len
        self.rand_shift = shift

    def collect_paths(self, root):
        paths = {'input' : [],
                 'output': [],
                 'labels': []}

        with open(os.path.join(root, 'dataset_combined.txt'), 'r') as f:
            lines = f.readlines()
            for idx in self.indices['combined']:
                inp, out, label = lines[idx].strip().split('\t')
                inp = os.path.join(self.data_root, inp)
                out = os.path.join(self.data_root, out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))

        with open(os.path.join(root, 'dataset_reverb.txt'), 'r') as f:
            lines = f.readlines()
            for idx in self.indices['reverb']:
                inp, out, label = lines[idx].strip().split('\t')
                inp = os.path.join(self.data_root, inp)
                out = os.path.join(self.data_root, out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))

        with open(os.path.join(root, 'dataset_linear.txt'), 'r') as f:
            lines = f.readlines()
            for idx in self.indices['linear']:
                inp, out, label, noise = lines[idx].strip().split('\t')
                inp = os.path.join(self.data_root, f"{noise.strip()}_list", inp)
                out = os.path.join(self.data_root, f"{noise.strip()}_list", out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))

        with open(os.path.join(root, 'dataset_eq.txt'), 'r') as f:
            lines = f.readlines()
            for idx in self.indices['eq']:
                inp, out, label = lines[idx].strip().split('\t')
                inp = os.path.join(self.data_root, inp)
                out = os.path.join(self.data_root, out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))
    
        return paths

    def __len__(self):
        return len(self.paths['input'])
 
    def __getitem__(self, idx):
     
        inp_file = self.paths['input'][idx]
        out_file = self.paths['output'][idx]

        inp, i_sr = torchaudio.load(inp_file)
        out, o_sr = torchaudio.load(out_file)

        if self.resample:
            inp = F.resample(inp, orig_freq=i_sr, new_freq=self.resample)
            out = F.resample(out, orig_freq=o_sr, new_freq=self.resample)
    
        inp = inp[:, :min(inp.shape[-1], out.shape[-1], self.cut_len)]
        out = out[:, :min(inp.shape[-1], out.shape[-1], self.cut_len)]

        if inp.shape[-1] < self.cut_len: 
            pad = torch.zeros(1, self.cut_len - inp.shape[-1])
            inp = torch.cat([pad, inp], dim=-1)
            out = torch.cat([pad, out], dim=-1)

        inp = inp.reshape(-1)
        out = out.reshape(-1)

        if self.paths['labels'][idx] == 1:
            label = torch.tensor([0.0, 1.0])
        else:
            label = torch.tensor([1.0, 0.0])
            
        return inp, out, label
    
def load_data(root, path_root, batch_size, n_cpu, split_ratio=0.7, cut_len=40000, resample=False, parallel=False):
    torchaudio.set_audio_backend("sox_io")  # in linux
    
    train_indices = {'combined':[], 'reverb':[], 'linear':[], 'eq':[]}
    test_indices = {'combined':[], 'reverb':[], 'linear':[], 'eq':[]}
    #For reproducing results
    np.random.seed(0)
    for key in train_indices:
        with open(os.path.join(path_root, f'dataset_{key}.txt'), 'r') as f:
            num_lines = len(f.readlines())
            #train_indxs = [i for i in range(int(split_ratio * num_lines))]
            train_indxs = np.random.choice(num_lines, int(split_ratio * num_lines), replace=False)
            test_indxs = [i for i in range(num_lines) if i not in train_indxs]
            print(f"KEY:{key} | TRAIN:{len(train_indxs)} | VAL:{len(test_indxs)}")
        train_indices[key].extend(train_indxs)
        test_indices[key].extend(test_indxs)
    if resample:
        resample = 16000
    train_ds = JNDDataset(root, path_root, train_indices, cut_len=cut_len, resample=resample)
    test_ds = JNDDataset(root, path_root, test_indices, cut_len=cut_len, resample=resample)

    if parallel:
        train_dataset = DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(train_ds),
            drop_last=True,
            num_workers=n_cpu,
            #collate_fn=collate_fn,
        )
        test_dataset = DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(test_ds),
            drop_last=True,
            num_workers=n_cpu,
            #collate_fn=collate_fn,
        )
    else:
        train_dataset = DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            num_workers=n_cpu,
            #collate_fn=collate_fn,
        )
        test_dataset = DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            num_workers=n_cpu,
            #collate_fn=collate_fn,
        )

    return train_dataset, test_dataset
