import os
import torch
import numpy as np
import torchaudio
from model.CMGAN.actor import TSCNet
from dns_mos import ComputeScore
from data.dataset import load_data, NISQAPreferenceDataset
from utils import preprocess_batch
from speech_enh_env import SpeechEnhancementAgent
from tqdm import tqdm
import soundfile as sf


class DataSampler:
    def __init__(self, dataloader, model, save_dir, num_samples=100, K=25):
        self.model = model
        self.model.eval()
        self.K = K
        self.n = num_samples
        self.t_low = -15
        self.t_high = 15
        self.dl = dataloader
        self._iter_ = iter(self.dl)
        
        self.root = save_dir
        self.sample_dir = f"{save_dir}/enhanced"
        self.x_dir = f"{save_dir}/noisy"
        self.y_pos_dir = f"{save_dir}/ypos"
        self.y_neg_dir = f"{save_dir}/yneg"

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.x_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.y_pos_dir, exist_ok=True)
        os.makedirs(self.y_neg_dir, exist_ok=True)

        p808_model_path = "/users/PAS2301/kumar1109/DNS-Challenge/DNSMOS/DNSMOS/model_v8.onnx"
        primary_model_path = "/users/PAS2301/kumar1109/DNS-Challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
        self.dns_mos = ComputeScore(primary_model_path, p808_model_path)

        self.env = SpeechEnhancementAgent(n_fft=400,
                                          hop=100,
                                          gpu_id=0,
                                          args=None,
                                          reward_model=None)
        
    def sample_batch(self, batch):

        batch = preprocess_batch(batch, 
                                 n_fft=self.env.n_fft, 
                                 hop=self.env.hop, 
                                 gpu_id=self.env.gpu_id, 
                                 return_c=True, 
                                 model='cmgan')
    
        _, _, noisy, _, c = batch
        noisy = noisy.permute(0, 1, 3, 2)
        bs = noisy.shape[0]
        noisy = noisy.repeat(self.K, 1, 1, 1)
        c = c.repeat(self.K)
        with torch.no_grad():
            noisy_ref = noisy[:bs, ...]
            #Set evaluation to True to deactivate sampling layer.
            #This is our reference enhanced output
            self.model.evaluation = True
            ref_next_state, _, _ = self.model.get_action(noisy_ref)
            ref_est_audio = self.env.get_audio(ref_next_state)

            #Set evaluation to False to activate sampling layer.
            #These are the sampled enhanced outputs
            noisy_rl = noisy[bs:, ...]
            self.model.evaluation = False
            next_state, _, _ = self.model.get_action(noisy_rl)
            est_audio = self.env.get_audio(next_state)

        est_audio = torch.cat([ref_est_audio, est_audio], dim=0)
        return est_audio, c
    
    def get_best_audio(self, audios, c):
        #Get MOS scores for all sampled audios
        nmos = self.env.get_NISQA_MOS_reward(audios, c, PYPATH="~/.conda/envs/rlhf-se/bin/python")
        #dmos = self.env.get_DNS_MOS_reward(audios, c, PYPATH="~/.conda/envs/rlhf-se/bin/python")
        dmos = self.dns_mos.get_scores(audios, desired_fs=16000, gpu_id=0)

        #reference audios
        n0, d0 = nmos[0], dmos[0]

        nmos = nmos - n0
        dmos = dmos - d0

        #Mask out mos values so that only those values with both 
        #higher nmos and dmos are considered.
        mos_mask = ((nmos + dmos) > 0) * (nmos * dmos > 0)
        nmos = mos_mask * nmos
        dmos = mos_mask * dmos

        #Get the average point for filtered mos values
        navg = nmos.mean()
        davg = dmos.mean()
        angle_avg = torch.rad2deg(torch.atan(navg / davg))

        #Get the angle relative to the average values
        angle = torch.rad2deg(torch.atan(nmos / dmos)) - angle_avg

        #Mask out every mag values whose angle lies outside the thresholds
        mask_l = angle > self.t_low
        mask_h = angle < self.t_high
        mag = ((nmos ** 2 + dmos ** 2) ** 0.5)
        mag = mask_l * (mag * mask_h)

        #Return the audio with the biggest magnitude
        idx = torch.argmax(mag)  
        return (audios[idx], audios[0])
    
    def generate_samples(self):
         for _ in tqdm(range(self.n)):
            try:
                batch = next(self._iter_)
            except StopIteration as e:
                self._iter_ = iter(self.dl)
                batch = next(self._iter_)

            _, noisy, filenames = batch

            try:
                audios, c = self.sample_batch(batch)
            except ValueError as e:
                continue

            a_map = {}
            batchsize = noisy.shape[0]
            
            for i, fname in enumerate(filenames):
                audios_i = audios[i::batchsize, ...]
                c_i = c[i::batchsize]
                audios_i = audios_i / c_i[0]
                ypos, yneg = self.get_best_audio(audios_i, c_i)
                a_map[fname] = {
                    'x':noisy[i, ...],
                    'ypos':ypos,
                    'yneg':yneg
                }
                
                self.save(a_map)

    def generate_triplets(self):
        #Remove previous stored data
        #self.delete()

        #Generate new data
        self.generate_samples()

        ds = NISQAPreferenceDataset(root=self.root)
        dl = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=4,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            num_workers=1,
        )

        return dl

    def delete(self):
        for _dir_ in os.listdir(self.root):
            subdir = os.path.join(self.root, _dir_)
            for wav in os.listdir(subdir):
                wav = os.path.join(subdir, wav)
                os.remove(wav)

    def save(self, audio_map):
        for fname in audio_map.keys():
            #Save samples if they exist
            samples = audio_map[fname].get('samples', None)
            if samples is not None:
                s_dir = os.path.join(self.sample_dir, fname)
                os.makedirs(s_dir, exist_ok=True)
                for i, sample in enumerate(samples):
                    sample = sample.detach().cpu().numpy().reshape(-1)
                    s_path = os.path.join(s_dir, f"sample_{i}.wav")
                    sf.write(s_path, sample, 16000)
                
            else:
                #Save ypos and yneg
                ypos = audio_map[fname].get('ypos')
                yneg = audio_map[fname].get('yneg')

                ypos_path = os.path.join(self.y_pos_dir, f"{fname}.wav")
                yneg_path = os.path.join(self.y_neg_dir, f"{fname}.wav")

                ypos = ypos.detach().cpu().numpy().reshape(-1)
                yneg = yneg.detach().cpu().numpy().reshape(-1)

                sf.write(ypos_path, ypos, 16000)
                sf.write(yneg_path, yneg, 16000)  

            #Save noisy
            x = audio_map[fname].get('x', None)
            x = x.detach().cpu().numpy().reshape(-1)
            x_path = os.path.join(self.x_dir, f"{fname}.wav")
            sf.write(x_path, x, 16000) 
    
if __name__ == '__main__':

    pre_pt = "/users/PAS2301/kumar1109/CMGAN/src/best_ckpt/ckpt"
    model_pre = TSCNet(num_channel=64, num_features=201, gpu_id=0, eval=False)
    pre_checkpoint = torch.load(pre_pt, map_location=torch.device('cpu'))
    model_pre.load_state_dict(pre_checkpoint)
    model_pre = model_pre.to(0)

    ds, _ = load_data("/users/PAS2301/kumar1109/NISQA_Corpus", 
                      4, 1, 
                      32000, gpu = False)
    
    sampler = DataSampler(ds, 
                          model=model_pre, 
                          save_dir="/fs/scratch/PAS2301/kumar1109/NISQA_Corpus", 
                          K=15, num_samples=100)
    
    sampler.generate_samples()
