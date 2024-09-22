# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.CMGAN.actor import TSCNet
from model.MetricGAN.actor import Generator
from model.reward_model import RewardModel
from RLHF2 import PPO
from compute_metrics import compute_metrics
from torch.utils.data import DataLoader


import os
from data.dataset import load_data, MixturesDataset, mixture_collate_fn
import torch.nn.functional as F
import torch
from utils import preprocess_batch, freeze_layers, map_state_dict
import logging
from torchinfo import summary
import argparse
import wandb
import psutil
import numpy as np
import traceback
import copy

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from speech_enh_env import SpeechEnhancementAgent

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rv", "--root_vctk", type=str, required=True,
                        help="Root directory to Voicebank.")
    parser.add_argument("-rd", "--root_dns", type=str, required=False,
                        help="Root directory to DNS.")
    parser.add_argument("--exp", type=str, required=False, default='default', help="Experiment name.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for checkpoints. Will create one if doesn't exist")
    parser.add_argument("-pt", "--ckpt", type=str, required=False, default=None,
                        help="Path to saved checkpoint to fine-tune.")
    parser.add_argument("--reward_pt", type=str, required=False, default=None,
                        help="path to the reward model checkpoint.")
    parser.add_argument("--epochs", type=int, required=False, default=5,
                        help="No. of epochs to be trained.")
    parser.add_argument("--batchsize", type=int, required=False, default=4,
                        help="Training batchsize.")
    parser.add_argument("--accum_grad", type=int, required=False, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--gpu", action='store_true',
                        help="Set this flag for single gpu training.")
    parser.add_argument("--parallel", action='store_true',
                        help="Set this flag for parallel gpu training.")
    parser.add_argument("--model", type=str, default='cmgan',
                        help="Choose between (metricgan/cmgan).")
    parser.add_argument("--train_phase", action='store_true',
                        help="Phase is also finetuned using RL.")
    parser.add_argument("--scale_reward", action='store_true',
                        help="Scale rewards by a factor of 0.1.")
    parser.add_argument("--suffix", type=str, required=False, default='',
                        help="Save path suffix")
    parser.add_argument("--beta", type=float, default=0.0, required=False,
                        help="KL weight")
    parser.add_argument("--ep_per_episode", type=int, default=1, required=False,
                        help="No of epochs per episode.")
    parser.add_argument("--lmbda", type=float, default=0.0, required=False,
                        help="Supervised pretrainig loss weight for PPO.")
    parser.add_argument("--episode_steps", type=int, default=1, required=False,
                        help="No. of steps in episode to run for PPO")
    parser.add_argument("--loss", type=str, default='pg', 
                        help="Terms to be included in loss. Should be a comma separated string, eg 'pg, mse' ")
    parser.add_argument("--reward", type=str, default='pesq', 
                        help="Terms to be included in loss. Should be a comma separated string, eg 'pesq, kl, mse' ")
    parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
    
    parser.add_argument("--gamma", type=float, default=0.99, help="Reward discount factor")
    parser.add_argument("--tau", type=float, default=0.99, help="target critic soft update factor")
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                    "and dereverberation")
    return parser

wandb.login()

torch.manual_seed(0)

class Trainer:
    """
    Starting with reinforce algorithm.
    """
    def __init__(self, 
                 train_ds, 
                 test_ds, 
                 args, 
                 gpu_id):
        
        if args.model == 'cmgan':
            self.n_fft = 400
            self.hop = 100
        if args.model == 'metricgan':
            self.n_fft = 512
            self.hop = 257

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.ACCUM_GRAD = args.accum_grad
        print(f"Finetuning {args.model}")
        if args.model == 'metricgan':
            self.actor = Generator(causal=False, 
                                   gpu_id=gpu_id)
            self.expert = Generator(causal=False, 
                                    gpu_id=gpu_id)
        elif args.model == 'cmgan':
            self.actor = TSCNet(num_channel=64, 
                                num_features=self.n_fft // 2 + 1, 
                                gpu_id=gpu_id)
            self.expert = TSCNet(num_channel=64, 
                                num_features=self.n_fft // 2 + 1,
                                gpu_id=gpu_id)
            
        else:
            raise NotImplementedError
                
        expert_checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))

        try:
            if args.model == 'cmgan':
                self.actor.load_state_dict(expert_checkpoint['generator_state_dict']) 
                self.expert.load_state_dict(expert_checkpoint['generator_state_dict'])
                self.expert.eval()

            if args.model == 'metricgan':
                self.actor.load_state_dict(expert_checkpoint['generator']) 
                self.expert.load_state_dict(expert_checkpoint['generator'])
                self.expert.eval()

        except KeyError as e:
            self.actor.load_state_dict(expert_checkpoint)
            self.expert.load_state_dict(expert_checkpoint)
        
        #Set expert to eval and freeze all layers.
        self.expert = freeze_layers(self.expert, 'all')
        
        del expert_checkpoint 
        print(f"Loaded checkpoint stored at {args.ckpt}. Resuming training...") 
        
        self.reward_model = None
        if args.reward_pt is not None:
            print(f"Loading reward_model from {args.reward_pt}")
            self.reward_model = RewardModel(in_channels=2)
            reward_checkpoint = torch.load(args.reward_pt, map_location=torch.device('cpu'))
            self.reward_model.load_state_dict(reward_checkpoint)
            self.reward_model = freeze_layers(self.reward_model, 'all')
            self.reward_model.eval()
            print(f"Loaded reward model from {args.reward_pt}...")
            del reward_checkpoint
            
        
        #Freeze complex decoder and reward model
        if not args.train_phase:
            if args.model == 'cmgan':
                self.actor = freeze_layers(self.actor, ['dense_encoder', 'TSCB_1', 'complex_decoder'])

        if gpu_id is not None:
            self.actor = self.actor.to(gpu_id)
            if self.expert is not None:
                self.expert = self.expert.to(gpu_id)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(gpu_id)
            
        self.optimizer = torch.optim.AdamW(
            filter(lambda layer:layer.requires_grad, self.actor.parameters()), lr=args.init_lr
        )
        self.c_optimizer = None
        self.trainer = PPO(loader=self.train_ds,
                            init_model=self.expert, 
                            reward_model=self.reward_model, 
                            gpu_id=gpu_id, 
                            beta=args.beta,
                            eps=0.02,
                            lmbda=args.lmbda, 
                            discount=0.99,
                            warm_up_steps=30,
                            scale_rewards=args.scale_reward,
                            batchsize=args.batchsize, 
                            run_steps=args.episode_steps,
                            train_phase=args.train_phase,
                            loss_type=args.loss,
                            reward_type=args.reward,
                            accum_grad=args.accum_grad,
                            model=args.model,
                            env_params={'n_fft':self.n_fft,
                                        'hop':self.hop, 
                                        'args':args})
            
        self.gpu_id = gpu_id
        self.G = 0
        self.args = args
        
        wandb.init(project=args.exp, name=args.suffix)

    
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
        noisy_phase = None
        if self.args.model == 'metricgan':
            noisy_phase = c
            c = torch.ones(noisy.shape[0], 1).to(self.gpu_id)

        inp = noisy.permute(0, 1, 3, 2)

        #Forward pass through actor to get the action(mask)
        next_state, log_probs, _ = self.actor.get_action(inp)
        est_audio = self.trainer.env.get_audio(next_state)
    
        ref_log_probs, _ = self.trainer.init_model.get_action_prob(inp, next_state)
        if self.args.model == 'cmgan':
            ref_log_prob = ref_log_probs[0] + ref_log_probs[1]
            log_prob = log_probs[0] + log_probs[1]

        if self.args.model == 'metricgan':
            ref_log_prob = ref_log_probs
            log_prob = log_probs
    
        kl_penalty = torch.mean(log_prob - ref_log_prob, dim=[1, 2, 3]).detach()
        ratio = torch.exp(kl_penalty)
        kl_penalty = ratio.detach()
        
        #Get reward
        for i in range(inp.shape[0]):
            rl_state.append(est_audio[i, ...])
            C.append(c[i, ...])

        #Supervised 
        if self.args.model == 'cmgan':
            mb_enhanced_mag = torch.sqrt(next_state[0]**2 + next_state[1]**2)
            mb_clean_mag = torch.sqrt(clean[:, 0, :, :]**2 + clean[:, 1, :, :]**2).unsqueeze(1).permute(0, 1, 3, 2)
            print(f"clean:{mb_clean_mag.shape}, enh:{mb_enhanced_mag.shape}")
            mag_loss = ((mb_clean_mag - mb_enhanced_mag)**2).mean() 
            mb_enhanced = torch.cat(next_state, dim=1)
            print(f"clean:{clean.shape}, enh:{mb_enhanced.shape}")
            clean = clean.permute(0, 1, 3, 2)
            ri_loss = ((clean - mb_enhanced) ** 2).mean()
            supervised_loss = 0.7*mag_loss + 0.3*ri_loss

        if self.args.model == 'metricgan':
            mb_enhanced_mag = next_state['est_mag'].permute(0, 1, 3, 2)
            mb_clean_mag = clean
            supervised_loss = (mb_clean_mag - mb_enhanced_mag)**2

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
        
        kl_penalty = kl_penalty.reshape(-1, 1)
        supervised_loss = supervised_loss.reshape(-1, 1)
        mb_pesq = mb_pesq.reshape(-1, 1)

                  
        metrics['mse'] = supervised_loss.mean()
        metrics['kl_penalty'] = kl_penalty.mean()
        metrics['rl_state'] = rl_state
        metrics['C'] = C
        return metrics
    

    def run_validation(self, episode):
        #Run validation
        self.actor.eval()
        self.actor.evaluation = True
        self.trainer.init_model.eval()
        self.trainer.init_model.evaluation = True
        
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

        print(f"Running evaluation at episode:{episode}")
        STATE = []
        C = []
        num_batches = len(self.test_ds['pre'])
        with torch.no_grad():
            for i, batch in enumerate(self.test_ds['pre']):
                
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
        
        loss = val_metrics['mse']/num_batches
        kl = val_metrics['kl_penalty']/num_batches
        
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
            "val_KL":kl,
            "reward_model_score":val_metrics['reward_model_score']
        }) 
        print(f"Episode:{episode} | VAL_PESQ:{np.asarray(val_metrics['pesq']).mean()} | VAL_LOSS:{loss} | RM_SCORE: {val_metrics['reward_model_score']}")
                
        return loss, val_metrics['reward_model_score'], np.asarray(val_metrics["pesq"]).mean()

    def train_one_epoch(self, epoch):       
        loss, best_rm_score, best_pesq = 9999, 0, 0
        epochs_per_episode = self.args.ep_per_episode
        loss, rm_score, val_pesq = self.run_validation(0)
        run_validation_step = 250 // (epochs_per_episode * self.args.episode_steps)
        print(f"Run validation at every step:{run_validation_step}")
        episode_per_epoch = (len(self.train_ds['pre']) // (self.args.batchsize * self.ACCUM_GRAD)) + 1
        print(f"TRAIN:{len(self.train_ds['pre'])}")
        print(f"ACCUM_GRAD:{self.ACCUM_GRAD} BATCH:{self.args.batchsize}")
        print(f"EPISODES_PER_EPOCH:{episode_per_epoch}")
        for i in range(episode_per_epoch):
            #try:
                loss, batch_reward, pesq = self.trainer.run_episode(self.actor, (self.optimizer, self.c_optimizer), n_epochs=epochs_per_episode)
                    
                if loss is not None:
                    wandb.log({
                        "episode": (i+1) + ((epoch - 1) * episode_per_epoch),
                        "episode_avg_kl":batch_reward[0].item(),
                        "episodic_avg_r": batch_reward[2].item(),
                        "episodic_reward_model_score": batch_reward[1].item(),
                        "clip_loss":loss[0],
                        "pretrain_loss":loss[2],
                        "pg_loss":loss[1],
                        "train_pesq":pesq, 
                    })

                    print(f"Epoch:{epoch} | Episode:{i+1} | Return: {batch_reward[2].item()} | RM_SCORE: {batch_reward[1].item()}")
                
                if (i+1) % 10 == 0:
                #Run validation after each episode
                    loss, rm_score, val_pesq = self.run_validation((epoch-1) * episode_per_epoch + (i+1))
                    if val_pesq >= best_pesq or rm_score >= best_rm_score:
                        best_pesq = val_pesq
                        best_rm_score = max(best_rm_score, rm_score)
                        self.save(loss, rm_score, val_pesq, (epoch-1) * episode_per_epoch + (i+1))
                
            #except Exception as e:
            #    print(traceback.format_exc())
            #    continue

    def train(self):
        """
        Run epochs, collect validation results and save checkpoints. 
        """
        print("Start training...")
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch+1)
            

    def save(self, loss, mos, pesq, episode=None):
        if episode is None:
            episode = len(self.train_ds)            
        if self.gpu_id == 0:
            checkpoint_prefix = f"{self.args.exp}_pesq_{pesq}_nmos_{mos}_loss_{loss}_episode_{episode}.pt"
            path = os.path.join(self.args.output, f"{self.args.exp}_{self.args.suffix}", checkpoint_prefix)
            save_dict = {'actor_state_dict':self.actor.state_dict(), 
                        'optim_state_dict':self.optimizer.state_dict()
                        }
            torch.save(save_dict, path)
    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int, args):
    if args.parallel:
        ddp_setup(rank, world_size)
        if rank == 0:
            print(args)
            available_gpus = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            print(f"Available gpus:{available_gpus}")

        train_ds, test_ds = load_data(args.root, 
                                    1, 
                                    1, 
                                    args.cut_len,
                                    gpu = True)
    else:
        if args.gpu:
            gpu = True
        else:
            gpu = False
        
        train_pre, test_pre = load_data(args.root_vctk, 
                                    args.batchsize,
                                    1, 
                                    args.cut_len,
                                    gpu = False)
        
        train_rl = train_pre
        test_rl = test_pre
        if args.root_dns is not None:
            train = MixturesDataset(clean_file_list=os.path.join(args.root_dns,"clean_train.list"),
                                    noise_file_list=os.path.join(args.root_dns, "noise.list"))
            
            test = MixturesDataset(clean_file_list=os.path.join(args.root_dns,"clean_dev.list"),
                                    noise_file_list=os.path.join(args.root_dns, "noise.list"))
            
            train_rl = DataLoader(
                dataset=train,
                batch_size=args.batchsize,
                pin_memory=True,
                shuffle=True,
                drop_last=False,
                num_workers=1,
                collate_fn=mixture_collate_fn
            )

            test_rl = DataLoader(
                dataset=test,
                batch_size=args.batchsize,
                pin_memory=True,
                shuffle=True,
                drop_last=False,
                num_workers=1,
                collate_fn=mixture_collate_fn
            )

    train_ds = {'pre':train_pre, 'rl':train_rl}
    test_ds = {'pre':test_pre, 'rl':test_rl}

    pretrain=False
    if args.ckpt is not None:
        pretrain=True
    print(args)
    trainer = Trainer(train_ds, test_ds, args, rank)
    
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    ARGS = args().parse_args()

    output = f"{ARGS.output}/{ARGS.exp}_{ARGS.suffix}"
    os.makedirs(output, exist_ok=True)

    world_size = torch.cuda.device_count()
    print(f"World size:{world_size}")
    if ARGS.parallel:
        mp.spawn(main, args=(world_size, ARGS), nprocs=world_size)
    else:
        if ARGS.gpu:
            main(0, world_size, ARGS)
        else:
            main(None, world_size, ARGS)




