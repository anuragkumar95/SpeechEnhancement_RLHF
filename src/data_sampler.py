import os
import torch
import numpy as np
import torchaudio
from model.CMGAN.actor import TSCNet
from dns_mos import ComputeScore
from data.dataset import load_data, NISQAPreferenceDataset
from utils import preprocess_batch, freeze_layers
from speech_enh_env import SpeechEnhancementAgent
from tqdm import tqdm
import soundfile as sf
import pickle

class DataSampler:
    def __init__(self, dataloader, model, save_dir, num_samples=100, K=25, gpu_id=None):
        self.model = model
        self.model.eval()
        self.K = K
        self.n = num_samples
        self.t_low = -15
        self.t_high = 15
        self.dl = dataloader
        self._iter_ = iter(self.dl)
        self.gpu_id = gpu_id
        
        self.root = save_dir
        self.sample_dir = f"{save_dir}/enhanced"
        self.x_dir = f"{save_dir}/noisy"
        self.y_pos_dir = f"{save_dir}/ypos"
        self.y_neg_dir = f"{save_dir}/yneg"
        self.score_dir = f"{save_dir}/scores"

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.x_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.y_pos_dir, exist_ok=True)
        os.makedirs(self.y_neg_dir, exist_ok=True)
        os.makedirs(self.score_dir, exist_ok=True)

        #Set expert to eval and freeze all layers.
        self.model = freeze_layers(self.model, 'all')

        p808_model_path = "/users/PAS2301/kumar1109/DNS-Challenge/DNSMOS/DNSMOS/model_v8.onnx"
        primary_model_path = "/users/PAS2301/kumar1109/DNS-Challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
        self.dns_mos = ComputeScore(primary_model_path, p808_model_path)

        self.env = SpeechEnhancementAgent(n_fft=400,
                                          hop=100,
                                          gpu_id=gpu_id,
                                          args=None,
                                          reward_model=None)
        
        self.a_map = {}
        
    def load_expert_model(self, model):
        self.model = model

        #Set expert to eval and freeze all layers.
        self.model = freeze_layers(self.model, 'all')

        
    def sample_batch(self, batch):
        print(f"GPU_ID:{self.env.gpu_id}")
        _, _, noisy, _, c = batch
        noisy = noisy.permute(0, 1, 3, 2)
        bs = noisy.shape[0]
        noisy = noisy.repeat(self.K, 1, 1, 1)
        #noisy = torch.stack([noisy for i in range(self.K)], dim=0).squeeze(1)
        print(f"NOISY:{noisy.shape}")

        c = c.repeat(self.K)
        with torch.no_grad():
            noisy_ref = noisy[:bs, ...]
            #Set evaluation to True to deactivate sampling layer.
            #This is our reference enhanced output
            self.model.evaluation = True
            ref_next_state, _, ref_actions = self.model.get_action(noisy_ref)
            ref_est_audio = self.env.get_audio(ref_next_state)
            #print(f"Ref done...")

            #Set evaluation to False to activate sampling layer.
            #These are the sampled enhanced outputs
            noisy_rl = noisy[bs:, ...]
            self.model.evaluation = False
            next_state, _, actions = self.model.get_action(noisy_rl)
            est_audio = self.env.get_audio(next_state)
            #print(f"RL done...")

        est_audio = torch.cat([ref_est_audio, est_audio], dim=0)

        print(f"REF_STATES:{ref_actions[0].shape, ref_actions[1].shape}")
        print(f"Y_STATES:{actions[0].shape, actions[1].shape}")

        actions = (
            torch.cat([ref_actions[0], actions[0]], dim=0), 
            torch.cat([ref_actions[1], actions[1]], dim=0)
        )
        
        print(f"ACTIONS:{actions[0].shape, actions[1].shape}")

        return est_audio, c, actions
    
    def get_best_audio(self, audios, c):
        #Get MOS scores for all sampled audios
        nmos_orig = self.env.get_NISQA_MOS_reward(audios.clone(), c, PYPATH="~/.conda/envs/rlhf-se/bin/python")
        dmos_orig = self.dns_mos.get_scores(audios.clone(), desired_fs=16000, gpu_id=self.gpu_id)

        #reference audios
        n0, d0 = nmos_orig[0], dmos_orig[0]

        nmos = nmos_orig - n0
        dmos = dmos_orig - d0

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
        print(f"Best audio index:{idx}")
        return (dmos_orig, nmos_orig), idx
    
    def generate_samples(self):
         for _ in tqdm(range(self.n)):
            try:
                batch = next(self._iter_)
            except StopIteration as e:
                self._iter_ = iter(self.dl)
                batch = next(self._iter_)

            _, noisy, filenames = batch
            batch = preprocess_batch(batch, 
                                     n_fft=self.env.n_fft, 
                                     hop=self.env.hop, 
                                     gpu_id=self.env.gpu_id, 
                                     return_c=True, 
                                     model='cmgan')
    
            _, _, noisy, _, c = batch
            try:
                audios, c, actions = self.sample_batch(batch)
            except ValueError as e:
                continue
            
            batchsize = noisy.shape[0]
          
            for i, fname in enumerate(filenames):
                audios_i = audios[i::batchsize, ...]
                c_i = c[i::batchsize]
                audios_i = audios_i / c_i[0]
                
                #Get best index
                scores, idx = self.get_best_audio(audios_i, c_i)

                #Collect a_pos and a_neg
                ypos = (
                    actions[0][idx, ...].detach().cpu(), 
                    actions[1][idx, ...].detach().cpu()
                )

                yneg = (
                    actions[0][0, ...].detach().cpu(), 
                    actions[0][idx, ...].detach().cpu()
                )

                scores = (
                    scores[0].detach().cpu(),
                    scores[1].detach().cpu(),
                )
                
                self.a_map[fname] = {
                    'x':noisy[i, ...].permute(0, 2, 1).detach().cpu(),
                    'ypos':ypos,
                    'yneg':yneg,
                    'scores':scores
                }

    def generate_triplets(self):
        #Remove previous stored data
        self.reset()

        #Generate new data
        print(f"Generating {self.n} triplets")
        self.generate_samples()

        ds = NISQAPreferenceDataset(data=self.a_map)
        dl = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=4,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            num_workers=1,
        )

        return dl

    def reset(self):
        self.a_map = {}
    
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
