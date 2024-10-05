import os
import torch
import numpy as np
from model.CMGAN.actor import TSCNet
from data.dataset import load_data
from utils import preprocess_batch
from speech_enh_env import SpeechEnhancementAgent
from tqdm import tqdm
import soundfile as sf


class DataSampler:
    def __init__(self, root, model, env, save_dir, K=25, cut_len=32000):
        self.model = model
        self.model.eval()
        self.env = env
        self.K = K
        self.t_low = -15
        self.t_high = 15
        self.dataloader, _ = load_data(root, 4, 1, cut_len, gpu = False)
        self.save_dir = save_dir
        os.makedirs(f"{self.save_dir}", exist_ok=True)
        os.makedirs(f"{self.save_dir}/noisy", exist_ok=True)
        os.makedirs(f"{self.save_dir}/ypos", exist_ok=True)
        os.makedirs(f"{self.save_dir}/yneg", exist_ok=True)
        
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
            self.model.evaluation = False
            next_state, _, _ = self.model.get_action(noisy[bs:, ...])
            est_audio = self.env.get_audio(next_state)

        est_audio = torch.cat([ref_est_audio, est_audio], dim=0)
        return est_audio, c
    
    def get_best_audio(self, audios, c):
        #Get MOS scores for all sampled audios
        nmos = self.env.get_NISQA_MOS_reward(audios, c)
        dmos = self.env.get_DNS_MOS_reward(audios, c)

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
        angle_avg = np.rad2deg(np.arctan(navg / davg))

        #Get the angle relative to the average values
        angle = torch.rad2deg(torch.atan(nmos / dmos)) - angle_avg

        #Mask out every mag values whose angle lies outside the thresholds
        mask_l = angle > self.t_low
        mask_h = angle < self.t_high
        mag = ((nmos ** 2 + dmos ** 2) ** 0.5)
        mag = mask_l * (mag * mask_h)

        #Return the audio with the biggest magnitude
        idx = torch.argmax(mag)  
        return audios[idx]
    
    def generate_triplets(self):

        print(f"Generating triplets with current best checkpoint...")

        for batch in tqdm(self.dataloader):
            _, noisy, filenames = batch
            audios, c = self.sample_batch(batch)

            batchsize = noisy.shape[0]
            y_pos = []
            y_neg = []
            for k in range(batchsize):
                audios_i = audios[k::batchsize, ...]
                c_i = c[k::batchsize]

                y_pos_i = self.get_best_audio(audios_i, c_i) / c_i[0]
                y_neg_i = audios_i[0, ...] / c_i[0]
                
                y_pos.append(y_pos_i)
                y_neg.append(y_neg_i)

            self.save(noisy, y_pos, y_neg, filenames)

    def save(self, x, ypos, yneg, filenames):
        for fname in filenames:
            x_path = os.path.join(self.save_dir, 'noisy', fname)
            ypos_path = os.path.join(self.save_dir, 'ypos', fname)
            yneg_path = os.path.join(self.save_dir, 'yneg', fname)
            
            x = x.detach().cpu().numpy().reshape(-1)
            ypos = ypos.detach().cpu().numpy().reshape(-1)
            yneg = yneg.detach().cpu().numpy().reshape(-1)

            sf.write(x_path, x, 16000)
            sf.write(ypos_path, ypos, 16000)
            sf.write(yneg_path, yneg, 16000)


    
if __name__ == '__main__':

    pre_pt = "/users/PAS2301/kumar1109/CMGAN/src/best_ckpt/ckpt"
    model_pre = TSCNet(num_channel=64, num_features=201, gpu_id=None, eval=False)
    pre_checkpoint = torch.load(pre_pt, map_location=torch.device('cpu'))
    model_pre.load_state_dict(pre_checkpoint)
    model_pre = model_pre.to(0)


    env = SpeechEnhancementAgent(n_fft=400,
                             hop=100,
                             gpu_id=0,
                             args=None,
                             reward_model=None)
    
    sampler = DataSampler(root="/users/PAS2301/kumar1109/NISQA_Corpus", 
                          model=model_pre, 
                          env=env, 
                          save_dir="/users/PAS2301/kumar1109/NISQA_Corpus", 
                          K=25, cut_len=32000)
    
    sampler.generate_triplets()
