import numpy as np
from model.actor import TSCNet, TSCNetSmall 
from model.reward_model import RewardModel
from natsort import natsorted
import os
from compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
from tqdm import tqdm
import pickle
import traceback
from speech_enh_env import SpeechEnhancementAgent

def run_enhancement_step(env, 
                         batch, 
                         actor,
                         lens,
                         file_id, 
                         save_dir,
                         save_track=True):
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
    action, _, _, _ = actor.get_action(inp)
    
    a_t = action
    
    #Apply action  to get the next state
    next_state = env.get_next_state(state=inp, 
                                    action=a_t)
    
    #Get reward
    r_state = env.get_RLHF_reward(state=next_state['noisy'].permute(0, 1, 3, 2), scale=False).mean()
    metrics['reward'] = r_state

    #Supervised loss
    mb_enhanced = next_state['noisy'].permute(0, 1, 3, 2)
    mb_enhanced_mag = torch.sqrt(mb_enhanced[:, 0, :, :]**2 + mb_enhanced[:, 1, :, :]**2)
    
    mb_clean_mag = torch.sqrt(clean[:, 0, :, :]**2 + clean[:, 1, :, :]**2)

    supervised_loss = ((clean - mb_enhanced) ** 2).mean() + ((mb_clean_mag - mb_enhanced_mag)**2).mean()
    metrics['mse'] = supervised_loss

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
        sf.write(saved_path, enh_audio/c, 16000)
    
    return metrics

def enhance_audios(model_pt, reward_pt, cutlen, noisy_dir, clean_dir, save_dir, gpu_id=None):
    
    #Initiate models
    model = TSCNet(num_channel=64, 
                   num_features=400 // 2 + 1,
                   distribution=None, 
                   gpu_id=gpu_id)
    
    reward_model = RewardModel(in_channels=2)
    
    #model_pt = "/users/PAS2301/kumar1109/CMGAN_RLHF/cmgan_big_ppo_ep5_steps10_beta1.1e-3_lmbda1.0_mvar0.01_cvar0.01_run6_forward2/cmgan_big_loss_0.12089217454195023_episode_75.pt"
    
    #Load cmgan model weights
    checkpoint = torch.load(model_pt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['actor_state_dict'])

    #Load reward model weights
    checkpoint = torch.load(reward_pt, map_location=torch.device('cpu'))
    reward_model.load_state_dict(checkpoint)

    if gpu_id is not None:
        model = model.to(gpu_id)
        reward_model = reward_model.to(gpu_id)

    model.eval()
    reward_model.eval()
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
    env = SpeechEnhancementAgent(n_fft=400,
                                 hop=100,
                                 gpu_id=gpu_id,
                                 args=None,
                                 reward_model=reward_model)

    with torch.no_grad():
        step = 0
        files = os.listdir(noisy_dir)
        for file in files:
            clean_file = os.path.join(clean_dir, file)
            noisy_file = os.path.join(noisy_dir, file)
            clean_ds, _ = torchaudio.load(clean_file)
            noisy_ds, _ = torchaudio.load(noisy_file)
            length = clean_ds.shape[-1]
            
            if length > cutlen:
                mb_size = clean_ds.shape[-1] // cutlen
                if clean_ds.shape[-1] % cutlen > 0:
                    mb_size = mb_size + 1
                end_idx = mb_size * cutlen
          
                cleans = []
                noises = []
                end = 0

                for i in range(mb_size-1):
                    st = i*cutlen
                    end = st + cutlen
                    cleans.append(clean_ds[:, st:end])
                    noises.append(noisy_ds[:, st:end])

                cleans.append(torch.cat([clean_ds[:, end:], clean_ds[:, :end_idx - clean_ds.shape[-1]]], dim=-1))
                noises.append(torch.cat([noisy_ds[:, end:], noisy_ds[:, :end_idx - noisy_ds.shape[-1]]], dim=-1))

                clean_ds = torch.stack(cleans, dim=0).squeeze(1)
                noisy_ds = torch.stack(noises, dim=0).squeeze(1)
            
            batch = (clean_ds, noisy_ds, length)
            batch = preprocess_batch(batch, gpu_id=None, return_c=True)
            
            #Run validation episode
            try:
                metrics = run_enhancement_step(env=env, 
                                               batch=batch, 
                                               actor=model, 
                                               lens=length,
                                               file_id=file,
                                               save_dir=save_dir,
                                               save_track=True)
                
                val_metrics['pesq'].extend(metrics['pesq'])
                val_metrics['csig'].extend(metrics['csig'])
                val_metrics['cbak'].extend(metrics['cbak'])
                val_metrics['covl'].extend(metrics['covl'])
                val_metrics['ssnr'].extend(metrics['ssnr'])
                val_metrics['stoi'].extend(metrics['stoi'])
                val_metrics['si-sdr'].extend(metrics['si-sdr'])
                val_metrics['mse'] += metrics['mse']
                val_metrics['reward'] += metrics['reward']

                msg = f"{file}: "
                for key in val_metrics:
                    if key not in ['mse', 'reward']:
                        msg += f"{key}:{val_metrics[key][-1]} | "
                    else:
                        msg += f"{key}:{metrics[key]} | "
                print(msg)
                print("="*50)
                step += 1

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
        
        msg = f"{file}: "
        for key in val_metrics:
            if key not in ['mse', 'reward']:
                msg += f"{key.capitalize()}:{val_metrics[key].mean()} | "
            else:
                msg += f"{key.capitalize()}:{metrics[key]} | "
        print(msg)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--model_path", type=str, default='./best_ckpt/ckpt_80',
                        help="the path where the model is saved")
    parser.add_argument("-rpt", "--reward_path", type=str, required=True,
                        help="the path where the model is saved")
    parser.add_argument("--test_dir", type=str, default='dir to your VCTK-DEMAND test dataset',
                        help="noisy tracks dir to be enhanced")
    parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
    parser.add_argument("--out_dist", action='store_true', help="toggle to test models that output normal dist.")
    parser.add_argument("--gpu", action='store_true', help="toggle to run models on gpu.")
    parser.add_argument("--pre", action='store_true', help="toggle to test pretrained models")
    parser.add_argument("--small", action='store_true', help="toggle to test small cmgan models")
    parser.add_argument("--cutlen", type=int, default=16 * 16000, help="length of signal to be passed to model. ")
    parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")

    args = parser.parse_args()


    noisy_dir = os.path.join(args.test_dir, "noisy")
    clean_dir = os.path.join(args.test_dir, "clean")
    
    if args.gpu:
        gpu_id = 0
    else:
        gpu_id = None

    enhance_audios(model_pt=args.model_path, 
                   reward_pt=args.reward_path, 
                   cutlen=args.cutlen, 
                   noisy_dir=noisy_dir, 
                   clean_dir=clean_dir, 
                   save_dir=args.save_dir, 
                   gpu_id=gpu_id)
