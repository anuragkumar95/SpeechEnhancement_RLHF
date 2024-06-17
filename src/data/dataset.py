import torch.utils.data
import torchaudio
import os
from utils import *
import random
from natsort import natsorted
import torchaudio.functional as F
from torch.distributions import Normal

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data.distributed import DistributedSampler

random.seed(123)

class MixturesDataset(torch.utils.data.Dataset):
    """
    This class generates a mixture dataset for SE model training.
    """
    def __init__(self, clean_file_list, noisy_file_list, cutlen=32000):
        self.clean_files = open(clean_file_list, 'r').readlines()
        self.noisy_files = open(noisy_file_list, 'r').readlines()
        self.snr_means = [-15, -10, -5, 0, 5, 10, 15]
        self.snr_std = 3.0
        self.cutlen = cutlen
    
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean = self.clean_files[idx]
        noise = np.random.choice(self.noisy_files)

        clean, c_sr = torchaudio.load(clean)
        noise, n_sr = torchaudio.load(noise)

        if c_sr != 16000:
            clean = F.resample(clean, orig_freq=c_sr, new_freq=16000)
        if n_sr != 16000:   
            noise = F.resample(noise, orig_freq=n_sr, new_freq=16000)

        if clean.shape[-1] > self.cutlen:
            clean = clean[:, :self.cutlen]

        if clean.shape[-1] < self.cutlen:
            while(clean.shape[-1] >= self.cutlen):
                clean = torch.cat([clean, clean], dim=-1)
            clean = clean[:, :self.cutlen]

        if noise.shape[-1] > self.cutlen:
            noise = noise[:, :self.cutlen]

        if noise.shape[-1] < self.cutlen:
            while(noise.shape[-1] >= self.cutlen):
                noise = torch.cat([noise, noise], dim=-1)
            noise = noise[:, :self.cutlen]

        snr_mean = np.random.choice(self.snr_means)
        snr = Normal(snr_mean, self.snr_std).sample()

        return clean, noise, snr
    

def mix_audios(clean, noise, snr):
        
    assert clean.shape[-1] == noise.shape[-1]

    #calculate the amount of noise to add to get a specific snr
    p_clean = (clean ** 2).mean().reshape(-1)
    p_noise = (noise ** 2).mean().reshape(-1)
    snr = snr.reshape(-1)

    p_ratio = p_clean / p_noise
    alpha = torch.sqrt(p_ratio / (10 ** (snr / 10)))
    signal = clean + (alpha * noise)
    
    return signal

def mixture_collate_fn(batch):

    clean = torch.tensor([i[0] for i in batch])
    noise = torch.tensor([i[1] for i in batch])
    snr = torch.tensor([i[2] for i in batch])

    mixed_signal = mix_audios(clean, noise, snr)

    return mixed_signal, clean


class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, "clean")
        self.noisy_dir = os.path.join(data_dir, "noisy")
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for _ in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len % length])
            noisy_ds_final.append(noisy_ds[: self.cut_len % length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start : wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start : wav_start + self.cut_len]

        return clean_ds, noisy_ds, length
    



    
def get_random_batch(ds, batch_size):
    random_index = np.random.choice(len(ds), batch_size, replace=False)
    batch = [ds.__getitem__(ind) for ind in random_index]
    return batch
    


def load_data(ds_dir, batch_size, n_cpu, cut_len, gpu=True):
    torchaudio.set_audio_backend("sox_io")  # in linux
    train_dir = os.path.join(ds_dir, "train")
    test_dir = os.path.join(ds_dir, "test")

    train_ds = DemandDataset(train_dir, cut_len)
    test_ds = DemandDataset(test_dir, cut_len)
    if gpu:
        train_dataset = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(train_ds),
            drop_last=False,
            num_workers=n_cpu,
        )
        test_dataset = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=64,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(test_ds),
            drop_last=False,
            num_workers=n_cpu,
        )
    else:
        train_dataset = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            num_workers=n_cpu,
        )
        test_dataset = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=64,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            num_workers=n_cpu,
        )

    return train_dataset, test_dataset