import numpy as np
from model.MetricGAN.actor import Generator
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
    
    clean_aud, clean, noisy, _, noisy_phase = batch
    inp = noisy.permute(0, 1, 3, 2)
    c = torch.ones(clean.shape[0], 1).to(env.gpu_id)

    #Forward pass through actor to get the action(mask)
    action, _, _, _ = actor.get_action(inp)

    if add_noise:
        #add gaussian noise to action
        m_mu = torch.zeros(action[0][1].shape)
        m_sigma = torch.ones(action[0][1].shape) * noise_std
        c_mu = torch.zeros(action[1].shape)
        c_sigma = torch.ones(action[1].shape) * noise_std
        m_dist = Normal(m_mu, m_sigma)
        c_dist = Normal(c_mu, c_sigma)

        m_noise = m_dist.sample().to(actor.gpu_id)
        c_noise = c_dist.sample().to(actor.gpu_id)
        
        (x, mask), comp_out = action
        mask += m_noise
        comp_out += c_noise
        
        action = ((x, mask), comp_out)
    
    #Apply action  to get the next state
    #next_state = env.get_next_state(state=inp, action=action)
    next_state = env.get_next_state(state=inp, 
                                    phase=noisy_phase, 
                                    action=action, 
                                    model="metricgan")
    
  

    clean_aud = clean_aud.reshape(-1)
    enh_audio = next_state['est_audio'].reshape(-1)

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

    if save_track:
        save_dir = os.path.join(save_dir, 'audios')
        os.makedirs(save_dir, exist_ok=True)
        saved_path = os.path.join(save_dir, file_id)

        est_audio = next_state['est_audio']/c
        est_audio = est_audio.reshape(-1)
        est_audio = est_audio.detach().cpu().numpy()

        est_audio = est_audio[:clean_aud.shape[-1]] 

        sf.write(saved_path, est_audio, 16000)
    
    return metrics

def enhance_audios(model_pt, cutlen, noisy_dir, save_dir, pre=False, clean_dir=None, gpu_id=None):
    
    #Initiate models
    model = Generator(causal=False, gpu_id=gpu_id)
    
    #Load cmgan model weights
    checkpoint = torch.load(model_pt, map_location=torch.device('cpu'))
    if pre:
        model.load_state_dict(checkpoint['generator'])
    else:
        model.load_state_dict(checkpoint['actor_state_dict'])

    if gpu_id is not None:
        model = model.to(gpu_id)

    model.eval()
    model.set_evaluation(True)

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
    env = SpeechEnhancementAgent(n_fft=512,
                                 hop=257,
                                 gpu_id=gpu_id,
                                 args=None,
                                 reward_model=None)

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
            #batch = preprocess_batch(batch, gpu_id=gpu_id, return_c=True)
            batch = preprocess_batch(batch, 
                                     n_fft=512, 
                                     hop=257, 
                                     gpu_id=gpu_id, 
                                     return_c=True,
                                     model='metricgan') 

            try:
                metrics = run_enhancement_step(env=env, 
                                               batch=batch, 
                                               actor=model, 
                                               lens=length,
                                               file_id=file,
                                               save_dir=save_dir,
                                               save_track=True)
                
            
                val_metrics['pesq'].append(metrics['pesq'])
                val_metrics['csig'].append(metrics['csig'])
                val_metrics['cbak'].append(metrics['cbak'])
                val_metrics['covl'].append(metrics['covl'])
                val_metrics['ssnr'].append(metrics['ssnr'])
                val_metrics['stoi'].append(metrics['stoi'])
                val_metrics['si-sdr'].append(metrics['si-sdr'])

                step += 1

                res_save_dir = os.path.join(save_dir, 'results')
                os.makedirs(res_save_dir, exist_ok=True)
                with open(os.path.join(res_save_dir, f'{file_id}_results.pickle'), 'wb') as f:
                    pickle.dump(metrics, f)


            except Exception as e:
                print(traceback.format_exc())
                continue
        
        
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
        'pesq':0,
        'csig':0,
        'cbak':0,
        'covl':0,
        'ssnr':0,
        'stoi':0,
        'si-sdr':0,
        'mse':0,
        'reward':0
    }
    
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
        print(file, values)
    
        metrics['pesq'] += values[0]
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
    parser.add_argument("--noisy_dir", type=str, default=None,
                        help="noisy tracks dir to be enhanced")
    parser.add_argument("--clean_dir", type=str, default=None,
                        help="clean tracks dir for metrics")
    parser.add_argument("--pre", action='store_true')
    parser.add_argument("--gpu", action='store_true', help="toggle to run models on gpu.")
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
                       cutlen=args.cutlen, 
                       noisy_dir=noisy_dir, 
                       clean_dir=clean_dir, 
                       pre=args.pre,
                       save_dir=args.save_dir,
                       gpu_id=gpu_id)
