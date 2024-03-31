from model.actor import TSCNet
from model.reward_model import RewardModel
from model.critic import QNet
#from model.cmgan import TSCNet
from RLHF import REINFORCE, PPO
from torch.utils.data import DataLoader
import torchaudio
import soundfile as sf

import copy
import os
from data.dataset import load_data
from reward_model.src.dataset.dataset import PreferenceDataset
import torch.nn.functional as F
import torchaudio.functional as AF
import torch
from utils import preprocess_batch, power_compress, power_uncompress, batch_pesq, copy_weights, freeze_layers, original_pesq
import logging
from torchinfo import summary
import argparse
import wandb
import psutil
import numpy as np
import traceback
from tqdm import tqdm
import torch
import os

import pickle
from speech_enh_env import SpeechEnhancementAgent

torch.manual_seed(123)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-jr", "--jndroot", type=str, required=False,
                        help="Root directory to JND Dataset.")
    parser.add_argument("-vr", "--vctkroot", type=str, required=False,
                        help="Root directory to VCTK Dataset.")
    parser.add_argument("-c", "--comp", type=str, required=False,
                        help="Root directory to JND Dataset comparision lists.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for results. Will create one if doesn't exist")
    parser.add_argument("-pt", "--ckpt", type=str, required=False, default=None,
                        help="Path to saved checkpoint to evaluate.")
    parser.add_argument("--pre", action="store_true")
    parser.add_argument("-rpt", "--reward_pt", type=str, required=False, default=None,
                        help="Path to saved rewardmodel checkpoint to evaluate.")
    parser.add_argument("--audio_dir", type=str, required=False, default=None,
                        help="directory to noisy audios to be enhanced. Required with flag --save_audios")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--save_actions", action='store_true', 
                        help="Flag to save actions")
    parser.add_argument("--save_specs", action='store_true', 
                        help="Flag to save enhanced spectograms")
    parser.add_argument("--save_scores", action='store_true', 
                        help="Flag to save critic scores")
    parser.add_argument("--save_pesq", action='store_true', 
                        help="Flag to save pesq values")
    parser.add_argument("--save_rewards", action='store_true',
                        help='Flag to save rewards from the reward model.')
    parser.add_argument("--save_audios", action='store_true',
                        help='Flag to save enhanced audios from the model.')
    return parser

class EvalModel:
    def __init__(self, modes, save_path, pre, args, model_pt=None, reward_pt=None, gpu_id=None):
        self.modes = modes
        self.n_fft = 400
        self.hop = 100
        
        self.actor = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1,
                            distribution="Normal", 
                            gpu_id=gpu_id)
        
        self.critic = None
        self.reward_model = None

        if model_pt is not None:
            self.critic = QNet(ndf=16, in_channel=2, out_channel=1)
            checkpoint = torch.load(model_pt, map_location=torch.device('cpu'))
            if pre:
                self.actor.load_state_dict(checkpoint['generator_state_dict'])
            else:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                if args.save_scores:
                    self.critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f"Loaded checkpoint from {model_pt}...")

            if gpu_id is not None:
                self.actor = self.actor.to(gpu_id)
                self.critic = self.critic.to(gpu_id)

            self.actor.eval()
            self.critic.eval()

        if reward_pt is not None:
            self.reward_model = RewardModel(policy=copy.deepcopy(self.actor))
            reward_checkpoint = torch.load(reward_pt, map_location=torch.device('cpu'))
            self.reward_model.load_state_dict(reward_checkpoint)
            print(f"Loaded reward model from {reward_pt} ... ")
            
            if gpu_id is not None:
                self.reward_model = self.reward_model.to(gpu_id)

            self.reward_model.eval()

        self.env = SpeechEnhancementAgent(n_fft=self.n_fft,
                                          hop=self.hop,
                                          gpu_id=gpu_id,
                                          args=None,
                                          reward_model=self.reward_model)

        self.save_path = save_path
        self.gpu_id = gpu_id

    def evaluate(self, dataset):
        
        with torch.no_grad():
            for mode in self.modes:
                save_path = f"{self.save_path}/{mode}"
                os.makedirs(save_path, exist_ok=True)
                for i, batch in enumerate(dataset):
                    
                    _, noisy_aud, _ = batch
                    #Preprocess batch
                    batch = preprocess_batch(batch, gpu_id=self.gpu_id)

                    cl_aud, clean, noisy, _ = batch
                    inp = noisy.permute(0, 1, 3, 2)

                    #Forward pass through actor to get the action(mask)
                    action, _, _, _ = self.actor.get_action(inp)

                    #Apply action  to get the next state
                    next_state = self.env.get_next_state(state=inp, 
                                                        action=action)
                    if mode == 'action':
                        with open(os.path.join(save_path, f"action_{i}.pickle"), 'wb') as f:
                            action = (action[0].detach().cpu().numpy(), action[1].detach().cpu().numpy())
                            pickle.dump(action, f)
                        print(f"action_{i}.pickle saved in {save_path}")

                    if mode == 'spectogram':
                        with open(os.path.join(save_path, f"spec_{i}.pickle"), 'wb') as f:
                            spec = {
                                'enhanced': next_state['noisy'].detach().cpu().numpy(),
                                'noisy'   : noisy.detach().cpu().numpy(),
                                'clean'   : clean.detach().cpu().numpy()
                            }
                            pickle.dump(spec, f)
                        print(f"spec_{i}.pickle saved in {save_path}")

                    if mode == 'critic_score':
                        score_clean = self.critic(clean)
                        score_noisy = self.critic(noisy)
                        score_enhanced = self.critic(next_state['noisy'])
                        scores = {
                            'enhanced' : score_enhanced.detach().cpu().numpy(),
                            'noisy'    : score_noisy.detach().cpu().numpy(),
                            'clean'    : score_clean.detach().cpu().numpy()
                        }
                        with open(os.path.join(save_path, f"score_{i}.pickle"), 'wb') as f:
                            pickle.dump(scores, f)
                        print(f"score_{i}.pickle saved in {save_path}")

                    if mode == 'pesq':
                        enh_aud = next_state['est_audio'].detach().cpu().numpy()
                        
                        n_pesq, pesq_mask = batch_pesq(cl_aud.detach().cpu().numpy(), noisy_aud.detach().cpu().numpy())
                        n_pesq = (n_pesq * pesq_mask)

                        e_pesq, pesq_mask = batch_pesq(cl_aud.detach().cpu().numpy(), enh_aud)
                        e_pesq = (e_pesq * pesq_mask)

                        pesq = {
                            'noisy':original_pesq(n_pesq),
                            'enhanced':original_pesq(e_pesq),
                        }

                        with open(os.path.join(save_path, f"pesq_{i}.pickle"), 'wb') as f:
                            pickle.dump(pesq, f)
                        print(f"pesq_{i}.pickle saved in {save_path}")

                    if mode == 'rewards':
                        enhanced = next_state['noisy']
                    
                        noisy_reward = self.reward_model.get_reward(inp=noisy.permute(0, 1, 3, 2))
                        clean_reward = self.reward_model.get_reward(inp=clean)
                        enhanced_reward = self.reward_model.get_reward(inp=enhanced.permute(0, 1, 3, 2))

                        rewards = {
                            'noisy': noisy_reward.detach().cpu().numpy(),
                            'clean': clean_reward.detach().cpu().numpy(),
                            'enhanced':enhanced_reward.detach().cpu().numpy()
                        }

                        with open(os.path.join(save_path, f"reward_{i}.pickle"), 'wb') as f:
                            pickle.dump(rewards, f)
                        print(f"reward_{i}.pickle saved in {save_path}")

    def enhance_one_track(self, audio_path, saved_dir, cut_len, n_fft=400, hop=100):
        name = os.path.split(audio_path)[-1]
        noisy, sr = torchaudio.load(audio_path)
        assert sr == 16000
        noisy = noisy.cuda()

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy = torch.transpose(noisy, 0, 1)
        noisy = torch.transpose(noisy * c, 0, 1)

        length = noisy.size(-1)
        frame_num = int(np.ceil(length / 100))
        padded_len = frame_num * 100
        padding_len = padded_len - length
        noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
        if padded_len > cut_len:
            batch_size = int(np.ceil(padded_len / cut_len))
            while 100 % batch_size != 0:
                batch_size += 1
            noisy = torch.reshape(noisy, (batch_size, -1))
        
        
        noisy_spec = torch.stft(
            noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True
        )
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        est_real, est_imag, _ = self.actor(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).cuda(),
            onesided=True,
        )
        est_audio = est_audio / c
        est_audio = torch.flatten(est_audio)[:length].cpu().numpy()

        if len(est_audio) < length:
            pad = np.zeros((1, length - len(est_audio)))
            est_audio = est_audio.reshape(1, -1)
            print(est_audio.shape, pad.shape)
            est_audio = np.stack([est_audio, pad], axis=0)
            est_audio = est_audio.reshape(-1)

        assert len(est_audio) == length, f"est:{len(est_audio)}, inp:{length}"
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

        
    def enhance_audio(self, src_dir):

        save_path = f"{self.save_path}/audios"
        os.makedirs(save_path, exist_ok=True)
        with torch.no_grad():
            for audio in tqdm(os.listdir(src_dir)):
                noisy_path = os.path.join(src_dir, audio)
                self.enhance_one_track(noisy_path, save_path, 16000 * 10)
           

if __name__ == '__main__':
    ARGS = args().parse_args()
    
    modes = []
    if ARGS.save_actions:
        modes.append('action')
    if ARGS.save_specs:
        modes.append('spectogram')
    if ARGS.save_scores:
        modes.append('critic_score')
    if ARGS.save_pesq:
        modes.append('pesq')
    if ARGS.save_rewards:
        modes.append('rewards')
        
    eval = EvalModel(modes=modes, 
                    model_pt=ARGS.ckpt, 
                    reward_pt=ARGS.reward_pt,
                    save_path=ARGS.output, 
                    pre=ARGS.pre,
                    args=ARGS,
                    gpu_id=0)
    
    if ARGS.save_audios:
        eval.enhance_audio(src_dir=ARGS.audio_dir)
    
    else:
        """
        test_dataset = PreferenceDataset(jnd_root=ARGS.jndroot, 
                                        vctk_root=ARGS.vctkroot, 
                                        set="test", 
                                        comp=ARGS.comp,
                                        train_split=0.8, 
                                        resample=16000,
                                        enhance_model=None,
                                        env=None,
                                        gpu_id=None,  
                                        cutlen=40000)
        
        dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=ARGS.batchsize,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            num_workers=1,
        )


        """
        train_ds, test_ds = load_data(ARGS.vctkroot, 
                            ARGS.batchsize, 
                            1, 
                            40000,
                            gpu = False)

        eval.evaluate(test_ds)

                    
