import numpy as np
from model.actor import TSCNet, TSCNetSmall 
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


@torch.no_grad()
def enhance_one_track(
    model, env, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False
):
    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)
    assert sr == 16000
    noisy = noisy.cuda()

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    if length > cut_len:
        return None, None
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

    action, _, _, _ = model.get_action(noisy_spec)
    next_state = env.get_next_state(state=noisy_spec, 
                                    action=action)
    #est_real, est_imag = model(noisy_spec)

    #est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    #est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    #est_audio = torch.istft(
    #    est_spec_uncompress,
    #    n_fft,
    #    hop,
    #    window=torch.hamming_window(n_fft).cuda(),
    #    onesided=True,
    #)
    est_audio = next_state['noisy']
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length, f"est_audio:{len(est_audio)} ref_audio:{length}"
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length


def evaluation(model_path, noisy_dir, clean_dir, cutlen, save_tracks, saved_dir, pre, small, dist):
    n_fft = 400
    if not dist:
        dist = None
    else:
        dist = 'Normal'
    if small:
        model = TSCNetSmall(num_channel=64, 
                            num_features=n_fft // 2 + 1, 
                            distribution=dist,
                            gpu_id=0,
                            eval=True).cuda()
    else:
        model = TSCNet(num_channel=64, 
                        num_features=n_fft // 2 + 1, 
                        distribution=dist,
                        gpu_id=0, 
                        eval=True).cuda()
        
    env = SpeechEnhancementAgent(n_fft=n_fft,
                                 hop=100,
                                 gpu_id=None,
                                 args=None,
                                 reward_model=None)
    
    if pre:
        try:
            model.load_state_dict(torch.load(model_path)['generator_state_dict'])
        except KeyError as e:
            model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path)['actor_state_dict'])

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    
    model = model.eval()

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    results = {'pesq':[], 'ssnr':[], 'ssi-sdr':[], 'c_sig':[], 'file':[]}
    print(f"Parsed {num} audios in {noisy_dir}...")
    metrics_total = np.zeros(7)
    for audio in tqdm(audio_list):
        try:
            with torch.no_grad():
                noisy_path = os.path.join(noisy_dir, audio)
                clean_path = os.path.join(clean_dir, audio)
                #est_audio, length = enhance_one_track(
                #    model, noisy_path, saved_dir, 16000 * 10, n_fft, n_fft // 4, save_tracks
                #)
                est_audio, length = enhance_one_track(
                    model, env, noisy_path, saved_dir, cutlen, n_fft, n_fft // 4, save_tracks
                )
                if est_audio is not None:
                    clean_audio, sr = sf.read(clean_path)
                    assert sr == 16000
                    metrics = compute_metrics(clean_audio, est_audio, sr, 0)
                    metrics = np.array(metrics)
                    metrics_total += metrics
                    results['pesq'].append(metrics[0])
                    results['file'].append(audio)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            continue

    metrics_avg = metrics_total / num
    print(
        "pesq: ",
        metrics_avg[0],
        "csig: ",
        metrics_avg[1],
        "cbak: ",
        metrics_avg[2],
        "covl: ",
        metrics_avg[3],
        "ssnr: ",
        metrics_avg[4],
        "stoi: ",
        metrics_avg[5],
        "si-sdr: ",
        metrics_avg[6],
    )

    with open(os.path.join(saved_dir, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f)

'''
def run_validation_step(self, env, clean, noisy):
        """
        Runs a vlidation loop for a batch.
        Predict mask for each frame one at a time 
        and return pesq score of the enhances batch of 
        spectrograms.
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
        #print("Running validation...")

        clean_aud, clean, noisy, _ = batch
        inp = noisy.permute(0, 1, 3, 2)

        #Forward pass through actor to get the action(mask)
        action, _, _, _ = self.actor.get_action(inp)
        if self.expert is not None:
            exp_action, _, _, _ = self.expert.get_action(inp)

        if self.args.train_phase:
            a_t = action
        else:
            a_t = (action[0], exp_action[-1])
        
        #Apply action  to get the next state
        next_state = env.get_next_state(state=inp, 
                                        action=a_t)
        
        #Get reward
        r_state = self.trainer.env.get_RLHF_reward(state=next_state['noisy'].permute(0, 1, 3, 2), 
                                       scale=False).sum()
        metrics['reward'] = r_state

        #Supervised loss
        mb_enhanced = next_state['noisy'].permute(0, 1, 3, 2)
        mb_enhanced_mag = torch.sqrt(mb_enhanced[:, 0, :, :]**2 + mb_enhanced[:, 1, :, :]**2)
        
        mb_clean_mag = torch.sqrt(clean[:, 0, :, :]**2 + clean[:, 1, :, :]**2)

        supervised_loss = ((clean - mb_enhanced) ** 2).mean() + ((mb_clean_mag - mb_enhanced_mag)**2).mean()
        metrics['mse'] = supervised_loss

        #Calculate metrics
        #pesq, pesq_mask = batch_pesq(clean_aud.detach().cpu().numpy(), 
        #                             next_state['est_audio'].detach().cpu().numpy())
        
        for i in range(self.args.batchsize):
            values = compute_metrics(clean_aud[i, ...].detach().cpu().numpy(), 
                                     next_state['est_audio'][i, ...].detach().cpu().numpy(), 
                                     16000, 
                                     0)
            metrics['pesq'].append(values[0])
            metrics['csig'].append(values[1])
            metrics['cbak'].append(values[2])
            metrics['covl'].append(values[3])
            metrics['ssnr'].append(values[4])
            metrics['stoi'].append(values[5])
            metrics['si-sdr'].append(values[6])
        
        #Calculate NISQA mos
        """
        ds = NL.SpeechQualityDataset(df=next_state['est_audio'].detach().cpu().numpy(), data_dir=None)

        val_mos, _ = NL.predict_mos(self.nisqa.model, 
                                    ds = ds, 
                                    bs=self.args.batchsize, 
                                    dev=self.gpu_id, 
                                    num_workers=0)
        """
        return metrics
    


    def run_validation(self, episode):
        #Run validation
        self.actor.eval()
        if self.args.method == 'PPO':
            self.actor.set_evaluation(True)
            self.critic.eval()
        pesq = 0
        loss = 0
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
        num_batches = len(self.test_ds)
        with torch.no_grad():
            for i, batch in enumerate(self.test_ds):
                
                #Preprocess batch
                batch = preprocess_batch(batch, gpu_id=self.gpu_id)
                
                #Run validation episode
                try:
                    metrics = self.run_validation_step(self.trainer.env, batch)
                    val_metrics['pesq'].extend(metrics['pesq'])
                    val_metrics['csig'].extend(metrics['csig'])
                    val_metrics['cbak'].extend(metrics['cbak'])
                    val_metrics['covl'].extend(metrics['covl'])
                    val_metrics['ssnr'].extend(metrics['ssnr'])
                    val_metrics['stoi'].extend(metrics['stoi'])
                    val_metrics['si-sdr'].extend(metrics['si-sdr'])
                    val_metrics['mse'] += metrics['mse']
                    val_metrics['reward'] += metrics['reward']

                except Exception as e:
                    print(traceback.format_exc())
                    continue
        
        loss = val_metrics['mse']/num_batches
        reward = val_metrics['reward']/(num_batches * self.args.batchsize)
 
        wandb.log({ 
            "episode": episode, 
            "val_scaled_pesq":pesq,
            "val_pretrain_loss":loss,
            "val_pesq":np.asarray(val_metrics["pesq"]).mean(),
            "val_csig":np.asarray(val_metrics["csig"]).mean(),
            "val_cbak":np.asarray(val_metrics["cbak"]).mean(),
            "val_covl":np.asarray(val_metrics["covl"]).mean(),
            "val_ssnr":np.asarray(val_metrics["ssnr"]).mean(),
            "val_stoi":np.asarray(val_metrics["stoi"]).mean(),
            "val_si-sdr":np.asarray(val_metrics["si-sdr"]).mean(),
            "val_reward":reward
        }) 
        print(f"Episode:{episode} | VAL_PESQ:{np.asarray(val_metrics['pesq']).mean()} | VAL_LOSS:{loss} | REWARD: {reward}")
        
        self.actor.train()
        if self.args.method == 'PPO':
            self.actor.set_evaluation(False)
            self.critic.train()
        
        return loss
'''   


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./best_ckpt/ckpt_80',
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, default='dir to your VCTK-DEMAND test dataset',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--out_dist", action='store_true', help="toggle to test models that output normal dist.")
parser.add_argument("--pre", action='store_true', help="toggle to test pretrained models")
parser.add_argument("--small", action='store_true', help="toggle to test small cmgan models")
parser.add_argument("--cutlen", type=int, default=16 * 16000, help="length of signal to be passed to model. ")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")

args = parser.parse_args()


if __name__ == "__main__":
    noisy_dir = os.path.join(args.test_dir, "noisy")
    clean_dir = os.path.join(args.test_dir, "clean")
    evaluation(args.model_path, noisy_dir, clean_dir, args.cutlen, args.save_tracks, args.save_dir, args.pre, args.small, args.out_dist)
