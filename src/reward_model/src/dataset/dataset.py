# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
Created on 23rd Nov, 2023
"""

import torch
import itertools
import random
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F
import os
import numpy as np

from tqdm import tqdm
from ..utils import get_specs 


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data.distributed import DistributedSampler

class JNDDataset(Dataset):
    def __init__(self, root=None, path_root=None, data=None, indices=None, cut_len=40000, resample=False, shift=False):
        if root is not None:
            self.data_root = root
            self.indices = indices
            self.resample = resample
            self.paths = self.collect_paths(path_root)
            self.data = None

        if data is not None:
            self.data = data[indices, :]
        self.cut_len = cut_len
        self.rand_shift = shift

    def collect_paths(self, root):
        paths = {'input' : [],
                 'output': [],
                 'labels': []}

        if 'combined' in self.indices:
            with open(os.path.join(root, 'dataset_combined.txt'), 'r') as f:
                lines = f.readlines()
                for idx in self.indices['combined']:
                    inp, out, label = lines[idx].strip().split('\t')
                    inp = os.path.join(self.data_root, inp)
                    out = os.path.join(self.data_root, out)
                    paths['input'].append(inp)
                    paths['output'].append(out)
                    paths['labels'].append(int(label))

        if 'reverb' in self.indices:
            with open(os.path.join(root, 'dataset_reverb.txt'), 'r') as f:
                lines = f.readlines()
                for idx in self.indices['reverb']:
                    inp, out, label = lines[idx].strip().split('\t')
                    inp = os.path.join(self.data_root, inp)
                    out = os.path.join(self.data_root, out)
                    paths['input'].append(inp)
                    paths['output'].append(out)
                    paths['labels'].append(int(label))

        if 'linear' in self.indices:
            with open(os.path.join(root, 'dataset_linear.txt'), 'r') as f:
                lines = f.readlines()
                for idx in self.indices['linear']:
                    inp, out, label, noise = lines[idx].strip().split('\t')
                    inp = os.path.join(self.data_root, f"{noise.strip()}_list", inp)
                    out = os.path.join(self.data_root, f"{noise.strip()}_list", out)
                    paths['input'].append(inp)
                    paths['output'].append(out)
                    paths['labels'].append(int(label))

        if 'eq' in self.indices:
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
        if self.data is None:
            return len(self.paths['input'])
        else:
            return self.data.shape[0]
 
    def __getitem__(self, idx):
        if self.data is None:
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

            #if self.paths['labels'][idx] == 1:
            #    label = torch.tensor([0.0, 1.0])
            #else:
            #    label = torch.tensor([1.0, 0.0])

        
        else:
            inp = torch.FloatTensor(self.data[idx, 0]).reshape(1, -1)
            out = torch.FloatTensor(self.data[idx, 1]).reshape(1, -1)

            inp = inp[:, :min(inp.shape[-1], out.shape[-1], self.cut_len)]
            out = out[:, :min(inp.shape[-1], out.shape[-1], self.cut_len)]

            if inp.shape[-1] < self.cut_len: 
                pad = torch.zeros(1, self.cut_len - inp.shape[-1])
                inp = torch.cat([pad, inp], dim=-1)
                out = torch.cat([pad, out], dim=-1)

            inp = inp.reshape(-1)
            out = out.reshape(-1)

        ch = np.random.choice(10)
        if ch >= 5:
            label = torch.tensor([1.0, 0.0])
            return inp[:self.cut_len], out[:self.cut_len], label
        label = torch.tensor([0.0, 1.0])
        return out[:self.cut_len], inp[:self.cut_len], label

class PreferenceDataset(torch.utils.data.Dataset):
    """
    Combined VCTK and JND Dataset for preferance 
    reward model training. Only the linear dataset
    from the JND Dataset is considered. 
    """
    def __init__(self, 
                 jnd_root, 
                 vctk_root, 
                 comp, 
                 set, 
                 train_split=0.8, 
                 resample=None,
                 enhance_model=None,
                 env=None,
                 gpu_id=None, 
                 cutlen=40000):
        
        self.j_root = jnd_root
        self.v_root = vctk_root
        self.cutlen = cutlen
        self.resample = resample
        self.set = set
        self.comp = comp
        self.split = train_split
        self.gpu_id = gpu_id
        self.env = env
        if enhance_model is not None:
            self.model = enhance_model.cpu()
            self.model.eval()
        else:
            self.model = None
        self.paths = self.collect_paths() 

    def collect_paths(self):
        paths = {
            'ref':[],
            'per':[],
            'enh':[],
        }

        with open(os.path.join(self.comp, 'dataset_linear.txt'), 'r') as f:
            lines = f.readlines()
            split_index = int(len(lines) * self.split)
            if self.set == 'train':
                set_lines = lines[:split_index]
            else:
                set_lines = lines[split_index:]
            for line in set_lines:
                inp, out, _, noise = line.strip().split('\t')
                inp = os.path.join(self.j_root, f"{noise.strip()}_list", inp)
                out = os.path.join(self.j_root, f"{noise.strip()}_list", out)
                paths['ref'].append(inp)
                paths['per'].append(out)

        if self.set == 'train':
            vctk_clean_dir = os.path.join(self.v_root, "train", "clean")
            vctk_noisy_dir = os.path.join(self.v_root, "train", "noisy")
        else:
            vctk_clean_dir = os.path.join(self.v_root, "test", "clean")
            vctk_noisy_dir = os.path.join(self.v_root, "test", "noisy")

        for clean, noisy in zip(os.listdir(vctk_clean_dir), os.listdir(vctk_noisy_dir)):
            inp = os.path.join(vctk_clean_dir, clean)
            out = os.path.join(vctk_noisy_dir, noisy)
            paths['ref'].append(inp)
            paths['per'].append(out)


        return paths
    
    def __len__(self):
        return len(self.paths['ref'])
    
    def __getitem__(self, idx):
        inp_file = self.paths['ref'][idx]
        out_file = self.paths['per'][idx]

        inp, i_sr = torchaudio.load(inp_file)
        out, o_sr = torchaudio.load(out_file)

        if self.resample is not None:
            inp = F.resample(inp, orig_freq=i_sr, new_freq=self.resample)
            out = F.resample(out, orig_freq=o_sr, new_freq=self.resample)
    
        inp = inp[:, :min(inp.shape[-1], out.shape[-1], self.cutlen)]
        out = out[:, :min(inp.shape[-1], out.shape[-1], self.cutlen)]

        if inp.shape[-1] < self.cutlen: 
            pad = torch.zeros(1, self.cutlen - inp.shape[-1])
            inp = torch.cat([pad, inp], dim=-1)
            out = torch.cat([pad, out], dim=-1)
        
        if self.model is not None:
            with torch.no_grad():
                _, _, noisy_spec = get_specs(out, inp, None, 400, 100)
                noisy_spec = noisy_spec.permute(0, 1, 3, 2)
                #Forward pass through actor to get the action(mask)
                action, _, _ = self.model.get_action(noisy_spec)

                #Apply action  to get the next state
                next_state = self.env.get_next_state(state=noisy_spec, 
                                                     action=action)
                enhanced_aud = next_state['est_audio'].detach()
            inp = inp.reshape(-1)
            out = out.reshape(-1)
            enhanced_aud = enhanced_aud.reshape(-1)
            label = torch.tensor([1.0, 3.0, 2.0])
            return inp[:self.cutlen], out[:self.cutlen], enhanced_aud[:self.cutlen], label
        
        inp = inp.reshape(-1)
        out = out.reshape(-1)

        label = torch.tensor([1.0, 0.0])
        return inp[:self.cutlen], out[:self.cutlen], label
    



class HumanAlignedDataset(Dataset):
    """
    Ranking Dataset created using the NISQA MOS ratings.
    """
    def __init__(self,
                 mixture_dir,
                 noisy_dir, 
                 rank, 
                 mos_file, 
                 batchsize=4,
                 cutlen=40000):
        self.mixture_dir = mixture_dir
        self.noisy_dir = noisy_dir
        self.ranks = rank
        self.cutlen = cutlen
        self.bs = batchsize
        self.mos = self.map_mos(mos_file)
        if self.ranks.endswith('.ranks'):
            self.pairs = self.map_ranks_to_pairs()
            self.keys = list(self.pairs.keys())
            max_len = max([len(self.pairs[key]) for key in self.pairs])
            print(f"Max batchsize:{max_len}")
        else:
            self.pairs = self.get_pairs()
        
        
    def map_mos(self, mos_file):
        _map_ = {}
        with open(mos_file) as f:
            lines = f.readlines()
            for line in lines[1:]:
                f, m = line.split(',')[:2]
                _map_[f] = float(m)
        return _map_
    
    def get_pairs(self):
        PAIRS = []
        with open(self.ranks) as f:
            lines = f.readlines()
            for line in lines:
                p1, p2 = line.split()[:2]
                path1 = os.path.join(self.mixture_dir, p1)
                path2 = os.path.join(self.mixture_dir, p2)
                PAIRS.append((path1, path2))
        return PAIRS

        
    def map_ranks_to_pairs(self):
        PAIRS = {}
    
        with open(self.ranks, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                files = line.strip().split(' ')
                FILES=[]
                #Put them all in a single list
                for file in files:
                    root = file[:-len(".wav")]
                    if "enh" not in root:
                        _id_ = root.split('-')[0]
                    else:
                        _id_ = root[len("enh_"):]
                    #root = "_".join(file.split('_')[:2])
                    val = (file, os.path.join(self.mixture_dir, file), os.path.join(self.noisy_dir, f"{root}.wav"))
                    FILES.append(val)

                #Find all possible combination of pairs from the ranking list
                #Since files is sorted, generated pairs will always have preferred 
                #rank indexed 1st within the pair
                pairs = itertools.combinations(FILES, 2)
                done = False
                for ((id1, path1, inp), (id2, path2, _)) in pairs:
                    mos_p1 = self.mos[id1]
                    mos_p2 = self.mos[id2]
                    if abs(mos_p1 - mos_p2) > 0.25:
                        done = True
                        _id_ = "_".join(id1.split("_")[:2])
                        if _id_ not in PAIRS:
                            PAIRS[_id_] = []
                        PAIRS[_id_].append((path1, path2, inp))
                if not done:
                    pairs = list(itertools.combinations(FILES, 2))
                    idx = np.random.choice(len(pairs), min(self.bs, len(pairs)), replace=False)
                    rand_pairs = [pairs[i] for i in idx]
                    for ((id1, path1, inp), (id2, path2, _)) in rand_pairs:
                        _id_ = "_".join(id1.split("_")[:2])
                        if _id_ not in PAIRS:
                            PAIRS[_id_] = []
                        PAIRS[_id_].append((path1, path2, inp))
                
        return PAIRS
    
    def __len__(self):
        if self.ranks.endswith('.ranks'):
            return len(self.keys)
        if self.ranks.endswith('.pairs'):
            return len(self.pairs)
    
    def __getitem__(self, idx):
        
        if self.ranks.endswith('.ranks'):
            pairs = self.pairs[self.keys[idx]]

            if len(pairs) > self.bs:
                idx = np.random.choice(len(pairs), self.bs, replace=False)
                rand_pairs = [pairs[i] for i in idx]
            else:
                rand_pairs = pairs

            X1 = []
            X2 = []
            INP = []
            for (path_1, path_2, inp) in rand_pairs:
                path_1 = path_1.strip()
                path_2 = path_2.strip()
                inp = inp.strip()

                x_1, sr_1 = torchaudio.load(path_1)
                x_2, sr_2 = torchaudio.load(path_2)
                x_in, sr_in = torchaudio.load(inp)

                assert sr_1 == sr_2 == sr_in 

                if x_1.shape[-1] < self.cutlen: 
                    pad = torch.zeros(1, self.cutlen - x_1.shape[-1])
                    x_1 = torch.cat([pad, x_1], dim=-1)
                    x_2 = torch.cat([pad, x_2], dim=-1)
                    x_in = torch.cat([pad, x_in], dim=-1)
                
                else:
                    start_idx = random.randint(0, x_1.shape[-1] - self.cutlen)
                    x_1 = x_1[:, start_idx: start_idx + self.cutlen]
                    x_2 = x_2[:, start_idx: start_idx + self.cutlen]
                    x_in = x_in[:, start_idx: start_idx + self.cutlen]
            
                x_1 = x_1.reshape(-1)
                x_2 = x_2.reshape(-1)
                x_in = x_in.reshape(-1)
                X1.append(x_1)
                X2.append(x_2)
                INP.append(x_in)

            X1 = torch.stack(X1)
            X2 = torch.stack(X2)
            INP = torch.stack(INP)
    
            return X1, X2, INP
        
        if self.ranks.endswith('.pairs'):
            pair = self.pairs[idx]
            
            path_1, path_2 = pair
            path_1 = path_1.strip()
            path_2 = path_2.strip()

            x_1, sr_1 = torchaudio.load(path_1)
            x_2, sr_2 = torchaudio.load(path_2)
            
            assert sr_1 == sr_2 

            if x_1.shape[-1] < self.cutlen: 
                pad = torch.zeros(1, self.cutlen - x_1.shape[-1])
                x_1 = torch.cat([pad, x_1], dim=-1)
            
            if x_2.shape[-1] <= self.cutlen: 
                pad = torch.zeros(1, self.cutlen - x_2.shape[-1])
                x_2 = torch.cat([pad, x_2], dim=-1)
        
            if x_1.shape[-1] > self.cutlen:
                start_idx = random.randint(0, x_1.shape[-1] - self.cutlen)
                x_1 = x_1[:, start_idx: start_idx + self.cutlen]
                
            if x_2.shape[-1] > self.cutlen:
                start_idx = random.randint(0, x_2.shape[-1] - self.cutlen)
                x_2 = x_2[:, start_idx: start_idx + self.cutlen]
        
            x_1 = x_1.reshape(-1)
            x_2 = x_2.reshape(-1)

            return x_1, x_2, x_2




def load_data(root=None, 
              data=None, 
              path_root=None, 
              batch_size=4, 
              type=None,
              n_cpu=1, 
              split_ratio=0.7, 
              cut_len=40000, 
              resample=False, 
              parallel=False, 
              shuffle=False):
    torchaudio.set_audio_backend("sox_io")  # in linux
    #For reproducing results
    np.random.seed(4)
    if data is None:
        if type is not None:
            if type not in ['combined', 'reverb', 'linear', 'eq']:
                raise ValueError("Unknown type for JND. Set type to be one of ('combined', 'reverb', 'linear', 'eq')")
            else:
                train_indices = {type:[]}
                test_indices = {type:[]}
        else:
            train_indices = {'combined':[], 'reverb':[], 'linear':[], 'eq':[]}
            test_indices = {'combined':[], 'reverb':[], 'linear':[], 'eq':[]}
        
        for key in train_indices:
            with open(os.path.join(path_root, f'dataset_{key}.txt'), 'r') as f:
                num_lines = len(f.readlines())

                train_indxs = np.random.choice(num_lines, int(split_ratio * num_lines), replace=False)
                test_indxs = [i for i in range(num_lines) if i not in train_indxs]

                print(f"KEY:{key} | TRAIN:{len(train_indxs)} | VAL:{len(test_indxs)}")
            train_indices[key].extend(train_indxs)
            test_indices[key].extend(test_indxs)

        if resample:
            resample = 16000
        train_ds = JNDDataset(root=root, path_root=path_root, indices=train_indices, cut_len=cut_len, resample=resample)
        test_ds = JNDDataset(root=root, path_root=path_root, indices=test_indices, cut_len=cut_len, resample=resample)

    else:
        print(f"loading data from {data} ...")
        data = np.load(data, encoding='latin1', allow_pickle=True)
        np.random.shuffle(data)
        train_indices = [idx for idx in range(0, int(split_ratio * data.shape[0]))]
        test_indices = [idx for idx in range(int(split_ratio * data.shape[0]), data.shape[0])]
        train_ds = JNDDataset(data=data, indices=train_indices)
        test_ds = JNDDataset(data=data, indices=test_indices)

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
            shuffle=True,
            drop_last=False,
            num_workers=n_cpu,
            #collate_fn=collate_fn,
        )
        test_dataset = DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=False,
            num_workers=n_cpu,
            #collate_fn=collate_fn,
        )

    return train_dataset, test_dataset
