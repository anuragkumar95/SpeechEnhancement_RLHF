#-*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.CMGAN.actor import TSCNet
from data.dataset import load_data
from data_sampler import DataSampler
import torch.nn.functional as F
import torch
from utils import power_compress, preprocess_batch, freeze_layers

import argparse
import wandb
import numpy as np

import torch
import wandb
#import copy
from torch.distributions import Normal

from compute_metrics import compute_metrics

import torch.multiprocessing as mp
import torch.nn.functional as F

#torch.manual_seed(123)
"""
TODO:
1. Add wandb logs.
2. Add validation loop
3. Implement proper argparsing.
4. Implement checkpoint saving and loading if resume training. 
"""

class DPO:
    def __init__(self,
                 sft_model,
                 model,   
                 gpu_id=None, 
                 beta=0.2,
                 wandb=False):
        
        self.ref_model = sft_model
        self.model = model
        self.gpu_id = gpu_id 
        self.std = 1.0
        self.beta = beta
        self.wandb = wandb


    def get_logprob(self, mu, x):
        std = (torch.ones(mu.shape) * self.std).to(self.gpu_id)
        N = Normal(mu, std)
        x_logprob = N.log_prob(x)
        return x_logprob 

    def dpo_loss(self, x, ypos, yneg):

        ypos = ypos.permute(0, 1, 3, 2)
        yneg = yneg.permute(0, 1, 3, 2)

        with torch.no_grad():
            ref_mu = self.ref_model(x)
            ref_mu = torch.cat([ref_mu[0], ref_mu[1]], dim=1)
            ref_pos_logprob = torch.mean(self.get_logprob(ref_mu, ypos), dim=[1, 2, 3])
            ref_neg_logprob = torch.mean(self.get_logprob(ref_mu, yneg), dim=[1, 2, 3])
        
        y_mu = self.model(x)
        y_mu = torch.cat([y_mu[0], y_mu[1]], dim=1)

        y_pos_logprob = torch.mean(self.get_logprob(y_mu, ypos), dim=[1, 2, 3])
        y_neg_logprob = torch.mean(self.get_logprob(y_mu, yneg), dim=[1, 2, 3])

        ypos_relative_logps = y_pos_logprob - ref_pos_logprob
        yneg_relative_logps = y_neg_logprob - ref_neg_logprob

        print(f"SHAPES:{ypos_relative_logps.shape}, {yneg_relative_logps.shape}")
        print(f"REF:{ref_pos_logprob}, {ref_neg_logprob}")
        print(f"RL:{y_pos_logprob}, {y_neg_logprob}")
        print(f"REL_POS:{ypos_relative_logps}")
        print(f"REL_NEG:{yneg_relative_logps}")
        scores = self.beta * (ypos_relative_logps - yneg_relative_logps) 
        print(f"SCORES: {scores}, {scores.shape}")
        log_scores = F.logsigmoid(scores).mean()
        return -log_scores, (ypos_relative_logps.mean(), yneg_relative_logps.mean())

    
    def spec(self, noisy, ypos, yneg, n_fft=400, hop=100):
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy = torch.transpose(noisy, 0, 1)
        ypos = torch.transpose(ypos, 0, 1) 
        yneg = torch.transpose(yneg, 0, 1)
        noisy = torch.transpose(noisy * c, 0, 1)
        ypos = torch.transpose(ypos * c, 0, 1)
        yneg = torch.transpose(yneg * c, 0, 1)

        noisy_spec = torch.stft(
            noisy,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).to(self.gpu_id),
            onesided=True,
        )
        ypos_spec = torch.stft(
            ypos,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).to(self.gpu_id),
            onesided=True,
        )
        yneg_spec = torch.stft(
            yneg,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).to(self.gpu_id),
            onesided=True,
        )

        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        ypos_spec = power_compress(ypos_spec)
        yneg_spec = power_compress(yneg_spec)
        
        return noisy_spec, ypos_spec, yneg_spec


    def forward_step(self, x, ypos, yneg):
        if self.gpu_id is not None:
            x = x.to(self.gpu_id)
            ypos = ypos.to(self.gpu_id)
            yneg = yneg.to(self.gpu_id)

        x, ypos, yneg = self.spec(x, ypos, yneg)
        dpo_loss, (ypos_logps, yneg_logps) = self.dpo_loss(x, ypos, yneg)

        if self.wandb:
            wandb.log({
                'dpo_loss':dpo_loss,
                'yposlogps':ypos_logps,
                'yneglogps':yneg_logps
            })

        return dpo_loss


class DPOTrainer:
    def __init__(self,
                 train_ds,
                 test_ds, 
                 args, 
                 gpu_id):
        self.args = args
        self.actor = TSCNet(num_channel=64, 
                            num_features=self.args.n_fft // 2 + 1, 
                            gpu_id=gpu_id,
                            eval=True)
        self.expert = TSCNet(num_channel=64, 
                            num_features=self.args.n_fft // 2 + 1,
                            gpu_id=gpu_id,
                            eval=True)
        
        expert_checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))

        try:
            self.actor.load_state_dict(expert_checkpoint['generator_state_dict']) 
            self.expert.load_state_dict(expert_checkpoint['generator_state_dict'])

        except KeyError as e:
            self.actor.load_state_dict(expert_checkpoint)
            self.expert.load_state_dict(expert_checkpoint)
        
        #Set expert to eval and freeze all layers.
        self.expert = freeze_layers(self.expert, 'all')
        
        del expert_checkpoint 
        print(f"Loaded checkpoint stored at {args.ckpt}.")
        
        if gpu_id is not None:
            self.actor = self.actor.to(gpu_id)
            self.expert = self.expert.to(gpu_id)

        self.optimizer = torch.optim.AdamW(
            filter(lambda layer:layer.requires_grad, self.actor.parameters()), lr=args.init_lr
        )     

        self.DPO = DPO(sft_model=self.expert,
                       model=self.actor,   
                       gpu_id=gpu_id, 
                       beta=0.1, )
        
        self.data_sampler = DataSampler(dataloader=train_ds, 
                                        model=self.expert, 
                                        save_dir="/fs/scratch/PAS2301/kumar1109/VCTK", 
                                        K=25, 
                                        num_samples=3,
                                        gpu_id=gpu_id)
        
        if args.wandb:
            wandb.init(project=args.exp, name=args.suffix)
        
        self.DPO = DPO(sft_model=self.expert,
                       model=self.actor,   
                       gpu_id=gpu_id, 
                       beta=0.1,
                       wandb=args.wandb)
    '''
    def run_validation_step(self, batch):
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
        rl_state = []
        C = []


        #print("Running validation...")
        clean_aud, clean, noisy, _, c = batch
        
        inp = noisy.permute(0, 1, 3, 2)

        #Forward pass through actor to get the action(mask)
        next_state, log_probs, _ = self.actor.get_action(inp)
        est_audio = self.trainer.env.get_audio(next_state)
        
        #Get reward
        for i in range(inp.shape[0]):
            rl_state.append(est_audio[i, ...])
            C.append(c[i, ...])

        for i in range(clean_aud.shape[0]):
            #if i >= clean_aud.shape[0]:
            #    break
            values = compute_metrics(clean_aud[i, ...].detach().cpu().numpy(), 
                                     rl_state[i].detach().cpu().numpy(), 
                                     16000, 
                                     0)
            
            metrics['pesq'].append(values[0])
            metrics['csig'].append(values[1])
            metrics['cbak'].append(values[2])
            metrics['covl'].append(values[3])
            metrics['ssnr'].append(values[4])
            metrics['stoi'].append(values[5])
            metrics['si-sdr'].append(values[6])

        mb_pesq = torch.tensor(metrics['pesq']).to(self.gpu_id)
        mb_pesq = mb_pesq.reshape(-1, 1)
        metrics['rl_state'] = rl_state
        metrics['C'] = C
        return metrics
    

    def run_validation(self, epoch):
        #Run validation
        self.actor.evaluation = True
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
            'mse':0,
            'kl_penalty':0,
            'reward_model_score':0
        }

        print(f"Running evaluation at epoch:{epoch}")
        STATE = []
        C = []
        num_batches = len(self.test_ds)
        with torch.no_grad():
            for i, batch in enumerate(self.test_ds):
                
                #Preprocess batch
                batch = preprocess_batch(batch, 
                                         n_fft=self.n_fft, 
                                         hop=self.hop, 
                                         gpu_id=self.gpu_id, 
                                         return_c=True, 
                                         model=self.args.model)
                
                #Run validation episode
                try:
                    metrics = self.run_validation_step(batch)
                    val_metrics['pesq'].extend(metrics['pesq'])
                    val_metrics['csig'].extend(metrics['csig'])
                    val_metrics['cbak'].extend(metrics['cbak'])
                    val_metrics['covl'].extend(metrics['covl'])
                    val_metrics['ssnr'].extend(metrics['ssnr'])
                    val_metrics['stoi'].extend(metrics['stoi'])
                    val_metrics['si-sdr'].extend(metrics['si-sdr'])
                    val_metrics['mse'] += metrics['mse']
                    val_metrics['kl_penalty'] += metrics['kl_penalty']
                    STATE.extend(metrics['rl_state'])
                    C.extend(metrics['C'])

                    print(f"Batch:{i} | VAL_PESQ:{np.asarray(val_metrics['pesq']).mean()} | VAL_LOSS:{val_metrics['mse']}")
        

                except Exception as e:
                    print(traceback.format_exc())
                    continue
            
            #Get MOS reward
            rm_score = self.trainer.env.get_NISQA_MOS_reward(audios=STATE, Cs=C)
            val_metrics['reward_model_score'] = rm_score.mean()
        
        
        wandb.log({ 
            "epoch": epoch, 
            "val_scaled_pesq":pesq,
            "val_pretrain_loss":loss,
            "val_pesq":np.asarray(val_metrics["pesq"]).mean(),
            "val_csig":np.asarray(val_metrics["csig"]).mean(),
            "val_cbak":np.asarray(val_metrics["cbak"]).mean(),
            "val_covl":np.asarray(val_metrics["covl"]).mean(),
            "val_ssnr":np.asarray(val_metrics["ssnr"]).mean(),
            "val_stoi":np.asarray(val_metrics["stoi"]).mean(),
            "val_si-sdr":np.asarray(val_metrics["si-sdr"]).mean(),
            "val_KL":kl,
            "reward_model_score":val_metrics['reward_model_score']
        }) 
        print(f"Episode:{episode} | VAL_PESQ:{np.asarray(val_metrics['pesq']).mean()} | VAL_LOSS:{loss} | RM_SCORE: {val_metrics['reward_model_score']}")
                
        return loss, val_metrics['reward_model_score'], np.asarray(val_metrics["pesq"]).mean()


    '''
    def train(self):
        print("Start training...")
        train_dl = self.data_sampler.generate_triplets()
        self.actor.train()
        self.expert.train()
        for epoch in range(self.args.epochs):
            for step, batch in enumerate(train_dl):
                x, ypos, yneg = batch
                #Get DPO loss
                loss = self.DPO.forward_step(x, ypos, yneg)
                loss = loss / self.args.accum_grad
                if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
                    loss.backward()
                    print(f"STEP:{step}|DPO_LOSS:{loss}")
                    print("="*100)
                    #Update network
                    if (step+1) % self.args.accum_grad == 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()


if __name__ == '__main__':

    train_ds, test_ds = load_data("/users/PAS2301/kumar1109/speech-datasets/VoiceBank/", 
                                  4, 1, 
                                  32000, gpu = False)
    
    class Args:
        def __init__(self, batchsize, ckpt, n_fft, hop, gpu_id, init_lr, epochs, accum_grad, wandb=False):
            self.batchsize = batchsize
            self.ckpt = ckpt
            self.n_fft = n_fft
            self.hop = hop
            self.gpu_id = gpu_id
            self.epochs = epochs
            self.init_lr = init_lr
            self.accum_grad = accum_grad
            self.wandb = wandb

    args = Args(4, "/users/PAS2301/kumar1109/CMGAN/src/best_ckpt/ckpt", 400, 100, 0, 0.0001, 50, 1)
    
    trainer = DPOTrainer(train_ds, test_ds, args=args, gpu_id=0)
    trainer.train()

 
            
