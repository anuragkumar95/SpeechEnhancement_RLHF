import numpy as np
from model.CMGAN.actor import TSCNet
from model.reward_model import RewardModel
from natsort import natsorted
import os
from compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
from tqdm import tqdm
import json
import pickle
import traceback
from speech_enh_env import SpeechEnhancementAgent
from torch.distributions import Normal
from pathlib import Path

def run_enhancement_step(env, 
                         batch, 
                         actor,
                         lens,
                         file_id, 
                         save_dir,
                         save_metrics=True,
                         save_track=True,
                         add_noise=False,
                         noise_std=0.01):
    """
    Enhances one audio at a time.
    ARGS:
        env        : (Object) Instance of SpeechAgentEnv
        batch      : (Tuple) Tuple of (clean, noisy) audio pair
        actor      : (TSCNet) Instance to the CMGAN model.
        len        : (Int) Length of the audio.
        file_id    : (Str) Filename of the audio.
        save_dir   : (Str) Path to the save directory for enhanced audio tracks
        save_track : Set this flag to save tracks.

    """
    metrics = {
        'pesq':[],
        'csig':[],
        'cbak':[],
        'covl':[],
        'ssnr':[],
        'stoi':[],
        'si-sdr':[],
        'mse':0,
        'reward':0}
    
    clean_aud, clean, noisy, _, c = batch
    inp = noisy.permute(0, 1, 3, 2)

    #Forward pass through actor to get the action(mask)
    if add_noise:
        actor.evaluation = False
    next_state, _, _= actor.get_action(inp)
    enh_audio = env.get_audio(next_state)

    if save_metrics:
        #Supervised loss
        clean = clean.permute(0, 1, 3, 2)
        mb_enhanced = torch.cat(next_state, dim=1)
        mb_enhanced_mag = torch.sqrt(mb_enhanced[:, 0, :, :]**2 + mb_enhanced[:, 1, :, :]**2)
        mb_clean_mag = torch.sqrt(clean[:, 0, :, :]**2 + clean[:, 1, :, :]**2)

        print(f"mag:{mb_clean_mag.shape, mb_enhanced_mag.shape}, enh:{mb_enhanced.shape}, clean:{clean.shape}")

        mag_loss = ((mb_clean_mag - mb_enhanced_mag)**2).mean() 
        ri_loss = ((clean - mb_enhanced) ** 2).mean()
        supervised_loss = 0.7*mag_loss + 0.3*ri_loss
        clean_aud = clean_aud.reshape(-1)

        clean_aud = clean_aud[:lens].detach().cpu().numpy()
        enh_audio = enh_audio[:lens].detach().cpu().numpy()

        values = compute_metrics(clean_aud, 
                                enh_audio, 
                                16000, 
                                0)
    
        metrics['pesq'] = values[0]
        metrics['csig'] = values[1]
        metrics['cbak'] = values[2]
        metrics['covl'] = values[3]
        metrics['ssnr'] = values[4]
        metrics['stoi'] = values[5]
        metrics['si-sdr'] = values[6]
        metrics['mse'] = supervised_loss.mean().cpu().numpy()

    if save_track:
        save_dir = os.path.join(save_dir, 'audios')
        os.makedirs(save_dir, exist_ok=True)
        saved_path = os.path.join(save_dir, file_id)

        enh_audio = enh_audio/c.reshape(-1, 1)
        enh_audio = enh_audio.reshape(-1)
        enh_audio = enh_audio.detach().cpu().numpy()

        sf.write(saved_path, enh_audio, 16000)
    
    return metrics

def enhance_audios(model_pt, reward_pt, cutlen, noisy_dir, save_dir, clean_dir=None, pre=False, gpu_id=None):
    
    #Initiate models
    model = TSCNet(num_channel=64, 
                   num_features=400 // 2 + 1,
                   gpu_id=gpu_id)
    
    #reward_model = RewardModel(in_channels=2)
    
    #model_pt = "/users/PAS2301/kumar1109/CMGAN_RLHF/cmgan_big_ppo_ep5_steps10_beta1.1e-3_lmbda1.0_mvar0.01_cvar0.01_run6_forward2/cmgan_big_loss_0.12089217454195023_episode_75.pt"
    #model_pt = "~/CMGAN_RLHF/cmgan_big_ppo_ep5_steps5_lmbda1.0_mvar0.01_cvar0.01/cmgan_big_PESQ_3.4969236850738525_epoch_1_episode_440.pt"
    
    #Load cmgan model weights
    checkpoint = torch.load(model_pt, map_location=torch.device('cpu'))
    if pre:
        try:
            model.load_state_dict(checkpoint['generator_state_dict'])
        except KeyError as e:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['actor_state_dict'])

    #Load reward model weights
    #checkpoint = torch.load(reward_pt, map_location=torch.device('cpu'))
    #reward_model.load_state_dict(checkpoint)

    if gpu_id is not None:
        model = model.to(gpu_id)
        #reward_model = reward_model.to(gpu_id)

    model.eval()
    #reward_model.eval()
    model.evaluation = True

    val_metrics = {
        'pesq':[],
        'csig':[],
        'cbak':[],
        'covl':[],
        'ssnr':[],
        'stoi':[],
        'si-sdr':[],
        'reward':0,
        'mse':0
    }

    #Initiate speech environment
    env = SpeechEnhancementAgent(n_fft=400,
                                 hop=100,
                                 gpu_id=gpu_id,
                                 args=None,)
                                 #reward_model=reward_model)

    with torch.no_grad():
        step = 0
        files = os.listdir(noisy_dir)
        for file in tqdm(files):
            file_id = file[:-len('.wav')]
            noisy_file = os.path.join(noisy_dir, file)
            noisy_ds, _ = torchaudio.load(noisy_file)
            if noisy_ds.shape[0] > 1:
                noisy_ds = noisy_ds[0, :].reshape(1, -1)
            if clean_dir is not None:
                clean_file = os.path.join(clean_dir, file)
                clean_ds, _ = torchaudio.load(clean_file)
                if clean_ds.shape[0] > 1:
                    clean_ds = clean_ds[0, :].reshape(1, -1)
                length = clean_ds.shape[-1]
            else:
                clean_ds = None
                length = noisy_ds.shape[-1]
            
            if length > cutlen:
                if clean_ds is not None:
                    mb_size = clean_ds.shape[-1] // cutlen
                    if clean_ds.shape[-1] % cutlen > 0:
                        mb_size = mb_size + 1
                else:
                    mb_size = noisy_ds.shape[-1] // cutlen
                    if noisy_ds.shape[-1] % cutlen > 0:
                        mb_size = mb_size + 1
                end_idx = mb_size * cutlen
          
                cleans = []
                noises = []
                end = 0

                for i in range(mb_size-1):
                    st = i*cutlen
                    end = st + cutlen
                    if clean_ds is not None:
                        cleans.append(clean_ds[:, st:end])
                    noises.append(noisy_ds[:, st:end])
                
                if clean_ds is not None:
                    cleans.append(torch.cat([clean_ds[:, end:], clean_ds[:, :end_idx - clean_ds.shape[-1]]], dim=-1))
                    clean_ds = torch.stack(cleans, dim=0).squeeze(1)
                    clean_ds = clean_ds[:min(clean_ds.shape[0], 2), :]

                noises.append(torch.cat([noisy_ds[:, end:], noisy_ds[:, :end_idx - noisy_ds.shape[-1]]], dim=-1))
                noisy_ds = torch.stack(noises, dim=0).squeeze(1)
                noisy_ds = noisy_ds[:min(noisy_ds.shape[0], 2), :]
            
            batch = (clean_ds, noisy_ds, length)
            batch = preprocess_batch(batch, gpu_id=gpu_id, return_c=True)
            if clean_ds is None:
                save_metrics=False
            else:
                save_metrics=True
            try:
                metrics = run_enhancement_step(env=env, 
                                               batch=batch, 
                                               actor=model, 
                                               lens=length,
                                               file_id=file,
                                               save_metrics=save_metrics,
                                               save_dir=save_dir,
                                               save_track=True)
                
                if save_metrics:
                    val_metrics['pesq'].append(metrics['pesq'])
                    val_metrics['csig'].append(metrics['csig'])
                    val_metrics['cbak'].append(metrics['cbak'])
                    val_metrics['covl'].append(metrics['covl'])
                    val_metrics['ssnr'].append(metrics['ssnr'])
                    val_metrics['stoi'].append(metrics['stoi'])
                    val_metrics['si-sdr'].append(metrics['si-sdr'])
                    val_metrics['mse'] += metrics['mse']
                    val_metrics['reward'] += metrics['reward']

                    step += 1

                    res_save_dir = os.path.join(save_dir, 'results')
                    os.makedirs(res_save_dir, exist_ok=True)
                    with open(os.path.join(res_save_dir, f'{file_id}_results.pickle'), 'wb') as f:
                        pickle.dump(metrics, f)


            except Exception as e:
                print(traceback.format_exc())
                continue
        
        if save_metrics:
            val_metrics['mse'] = val_metrics['mse']/step
            val_metrics['reward'] = val_metrics['reward']/step
            val_metrics['pesq'] = np.asarray(val_metrics['pesq'])
            val_metrics['csig'] = np.asarray(val_metrics['csig'])
            val_metrics['cbak'] = np.asarray(val_metrics['cbak'])
            val_metrics['covl'] = np.asarray(val_metrics['covl'])
            val_metrics['ssnr'] = np.asarray(val_metrics['ssnr'])
            val_metrics['stoi'] = np.asarray(val_metrics['stoi'])
            val_metrics['si-sdr'] = np.asarray(val_metrics['si-sdr'])
            
            msg = ""
            for key in val_metrics:
                if key not in ['mse', 'reward']:
                    msg += f"{key.capitalize()}:{val_metrics[key].mean()} | "
                else:
                    msg += f"{key.capitalize()}:{metrics[key]} | "
            print(msg)

def compute_scores(clean_dir, enhance_dir, save_dir=None):

    metrics = {
        'pesq':[],
        'csig':[],
        'cbak':[],
        'covl':[],
        'ssnr':[],
        'stoi':[],
        'si-sdr':[],
        'mse':[],
        'reward':[]}
    
    num_files = len(os.listdir(enhance_dir))

    for file in tqdm(os.listdir(enhance_dir)):
        enh_file = os.path.join(enhance_dir, file)
        clean_file = os.path.join(clean_dir, file)
        file_id = Path(file).stem

        clean_aud, sr = torchaudio.load(clean_file)
        enh_audio, sr = torchaudio.load(enh_file) 

        values = compute_metrics(clean_aud.reshape(-1).cpu().numpy(), 
                                 enh_audio.reshape(-1).cpu().numpy(), 
                                 16000, 
                                 0)
    
        metrics['pesq'].append(values[0])
        metrics['csig'] += values[1]
        metrics['cbak'] += values[2]
        metrics['covl'] += values[3]
        metrics['ssnr'] += values[4]
        metrics['stoi'] += values[5]
        metrics['si-sdr'] += values[6]

        results = {key:0 for key in metrics}
        results['pesq'] = values[0]
        results['csig'] = values[1]
        results['cbak'] = values[2]
        results['covl'] = values[3]
        results['ssnr'] = values[4]
        results['stoi'] = values[5]
        results['si-sdr'] = values[6]

        if save_dir:
            res_save_dir = os.path.join(save_dir, 'results')
            os.makedirs(res_save_dir, exist_ok=True)
            with open(os.path.join(res_save_dir, f'{file_id}_results.pickle'), 'wb') as f:
                pickle.dump(results, f)

    for key in metrics:
        metrics[key] = metrics[key] / num_files

    msg = ""
    for key in metrics:
        msg += f"{key.capitalize()}:{metrics[key]} | "
       
    print(msg)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--model_path", type=str, default='./best_ckpt/ckpt_80',
                        help="the path where the model is saved")
    parser.add_argument("-rpt", "--reward_path", type=str, required=False,
                        help="the path where the model is saved")
    parser.add_argument("--noisy_dir", type=str, default=None,
                        help="noisy tracks dir to be enhanced")
    parser.add_argument("--clean_dir", type=str, default=None,
                        help="clean tracks dir for metrics")
    parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
    parser.add_argument("--out_dist", action='store_true', help="toggle to test models that output normal dist.")
    parser.add_argument("--gpu", action='store_true', help="toggle to run models on gpu.")
    parser.add_argument("--pre", action='store_true', help="toggle to test pretrained models")
    parser.add_argument("--small", action='store_true', help="toggle to test small cmgan models")
    parser.add_argument("--cutlen", type=int, default=16 * 16000, help="length of signal to be passed to model. ")
    parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")
    parser.add_argument("--enhance_dir", type=str, default=None, help="Path to enhanced_dir")
  

    args = parser.parse_args()


    #noisy_dir = os.path.join(args.test_dir, "noisy")
    #clean_dir = os.path.join(args.test_dir, "clean")
    noisy_dir = args.noisy_dir
    clean_dir = args.clean_dir
    
    if args.gpu:
        gpu_id = 0
    else:
        gpu_id = None

    if args.enhance_dir is not None:
        compute_scores(clean_dir, args.enhance_dir, args.save_dir)
    else:
        enhance_audios(model_pt=args.model_path, 
                       reward_pt=args.reward_path, 
                       cutlen=args.cutlen, 
                       noisy_dir=noisy_dir, 
                       clean_dir=clean_dir, 
                       save_dir=args.save_dir,
                       pre=args.pre, 
                       gpu_id=gpu_id)
