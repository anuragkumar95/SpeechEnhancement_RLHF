# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

from model.actor import TSCNet, TSCNetSmall
from model.critic import QNet, Critic
from model.reward_model import RewardModel
from RLHF import REINFORCE, PPO
#import NISQA.nisqa.NISQA_lib as NL
#from NISQA.nisqa.NISQA_model import nisqaModel
from compute_metrics import compute_metrics
from torch.utils.data import DataLoader


import os
from data.dataset import load_data, MixturesDataset, mixture_collate_fn
import torch.nn.functional as F
import torch
from utils import preprocess_batch, power_compress, power_uncompress, batch_pesq, copy_weights, freeze_layers, original_pesq
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
    parser.add_argument("--out_dist", action='store_true',
                        help="If GAN learns a distribution.")
    parser.add_argument("--small", action='store_true',
                        help="Finetuning small GAN model.")
    parser.add_argument("--train_phase", action='store_true',
                        help="Phase is also finetuned using RL.")
    parser.add_argument("--scale_reward", action='store_true',
                        help="Scale rewards by a factor of 0.1.")
    parser.add_argument("--suffix", type=str, required=False, default='',
                        help="Save path suffix")
    parser.add_argument("--method", type=str, default='reinforce', required=False,
                        help="RL Algo to run. Choose between (reinforce/PPO)")
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

torch.manual_seed(111)

class Trainer:
    """
    Starting with reinforce algorithm.
    """
    def __init__(self, 
                 train_ds, 
                 test_ds, 
                 args, 
                 gpu_id):
        
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.ACCUM_GRAD = args.accum_grad
        dist = None

        if args.out_dist:
            dist = 'Normal'
        if args.small:
            self.actor = TSCNetSmall(num_channel=64, 
                                num_features=self.n_fft // 2 + 1,
                                gpu_id=gpu_id)
        else:
            self.actor = TSCNet(num_channel=64, 
                                num_features=self.n_fft // 2 + 1, 
                                gpu_id=gpu_id)
        
        self.expert = None
        if args.ckpt is not None:
            if args.small:
                self.expert = TSCNetSmall(num_channel=64, 
                                num_features=self.n_fft // 2 + 1, 
                                gpu_id=gpu_id)
            else:
                self.expert = TSCNet(num_channel=64, 
                                num_features=self.n_fft // 2 + 1,
                                gpu_id=gpu_id)
            cmgan_expert_checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
            try:
                self.actor.load_state_dict(cmgan_expert_checkpoint['generator_state_dict']) 
                self.expert.load_state_dict(cmgan_expert_checkpoint['generator_state_dict'])
              
            except KeyError as e:
                self.actor.load_state_dict(cmgan_expert_checkpoint)
                self.expert.load_state_dict(cmgan_expert_checkpoint)
           
            #Set expert to eval and freeze all layers.
            self.expert = freeze_layers(self.expert, 'all')
            
            del cmgan_expert_checkpoint 
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
            self.actor = freeze_layers(self.actor, ['dense_encoder', 'TSCB_1', 'complex_decoder'])

        if gpu_id is not None:
            self.actor = self.actor.to(gpu_id)
            if self.expert is not None:
                self.expert = self.expert.to(gpu_id)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(gpu_id)

        if args.method == 'reinforce':
            
            self.optimizer = torch.optim.AdamW(
                filter(lambda layer:layer.requires_grad,self.actor.parameters()), lr=args.init_lr
            )
            
            self.trainer = REINFORCE(loader=self.train_ds,  
                                     init_model=self.expert,
                                     discount=1.0,
                                     episode_len=args.episode_steps,
                                     train_phase=args.train_phase,
                                     reward_model=self.reward_model,
                                     gpu_id=gpu_id,
                                     beta=args.beta,
                                     lmbda=args.lmbda,
                                     loss_type=args.loss,
                                     reward_type=args.reward,
                                     batchsize=args.batchsize,
                                     env_params={'n_fft':400,
                                                 'hop':100, 
                                                 'args':args})
            
        if args.method == 'PPO':
            #self.critic = QNet(ndf=16, in_channel=2, out_channel=1)
            #self.critic = Critic(in_channels=2)
            #reward_checkpoint = torch.load(args.reward_pt, map_location=torch.device('cpu'))
            #self.critic.load_state_dict(reward_checkpoint)
            #self.critic = self.critic.to(gpu_id)

            #Initialize critic 
            self.optimizer = torch.optim.AdamW(
                filter(lambda layer:layer.requires_grad, self.actor.parameters()), lr=args.init_lr
            )
            #self.c_optimizer = torch.optim.AdamW(
            #    filter(lambda layer:layer.requires_grad, self.critic.parameters()), lr=1e-05
            #)
            self.c_optimizer = None

            self.trainer = PPO(loader=self.train_ds,
                               init_model=self.expert, 
                               reward_model=self.reward_model, 
                               gpu_id=gpu_id, 
                               beta=args.beta,
                               eps=0.02,
                               val_coef=1.0,
                               en_coef=0,
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
                               env_params={'n_fft':400,
                                            'hop':100, 
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
        #print("Running validation...")
        clean_aud, clean, noisy, _, c = batch
        inp = noisy.permute(0, 1, 3, 2)

        #Forward pass through actor to get the action(mask)
        action, log_probs, _, _ = self.actor.get_action(inp)
        
        if self.expert is not None:
            ref_log_probs, _ = self.trainer.init_model.get_action_prob(inp, action)
            ref_log_prob = ref_log_probs[0] + ref_log_probs[1][:, 0, :, :].permute(0, 2, 1) + ref_log_probs[1][:, 1, :, :].permute(0, 2, 1)

        log_prob = log_probs[0] + log_probs[1][:, 0, :, :].permute(0, 2, 1) + log_probs[1][:, 1, :, :].permute(0, 2, 1)

        if ref_log_prob is not None:
            kl_penalty = torch.mean(log_prob - ref_log_prob, dim=[1, 2]).detach()
            ratio = torch.exp(kl_penalty)
            kl_penalty = ratio.detach()
    
        #Apply action  to get the next state
        next_state = self.trainer.env.get_next_state(state=inp, action=action)
        
        #Get reward
        #r_state = self.trainer.env.get_RLHF_reward(state=next_state['noisy'].permute(0, 1, 3, 2), scale=False)
        r_state = self.trainer.env.get_NISQA_MOS_reward(audio=next_state['est_audio'], c=c)

        #Supervised loss
        mb_enhanced = next_state['noisy'].permute(0, 1, 3, 2)
        mb_enhanced_mag = torch.sqrt(mb_enhanced[:, 0, :, :]**2 + mb_enhanced[:, 1, :, :]**2)
        
        mb_clean_mag = torch.sqrt(clean[:, 0, :, :]**2 + clean[:, 1, :, :]**2)

        mag_loss = (mb_clean_mag - mb_enhanced_mag)**2
        ri_loss = (clean - mb_enhanced) ** 2
        supervised_loss = 0.3 * torch.mean(ri_loss, dim=[1, 2, 3]) + 0.7 * torch.mean(mag_loss, dim=[1, 2])

        for i in range(64):
            if i >= clean_aud.shape[0]:
                break
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

        mb_pesq = torch.tensor(metrics['pesq']).to(self.gpu_id)
        
        kl_penalty = kl_penalty.reshape(-1, 1)
        supervised_loss = supervised_loss.reshape(-1, 1)
        mb_pesq = mb_pesq.reshape(-1, 1)

        reward = r_state - self.args.beta * kl_penalty - self.args.lmbda * (supervised_loss - mb_pesq)
                  
        metrics['mse'] = supervised_loss.mean()
        metrics['reward'] = reward.mean()
        metrics['kl_penalty'] = kl_penalty.mean()
        metrics['reward_model_score'] = r_state.mean()

        print(f"REWARD:{metrics['reward']} | RM_Score: {metrics['reward_model_score']} | KL: {metrics['kl_penalty']} | MSE: {metrics['mse']} | PESQ: {mb_pesq.mean()}")

        return metrics
    


    def run_validation(self, episode):
        #Run validation
        self.actor.eval()
        if self.args.method == 'PPO':
            self.actor.set_evaluation(True)
            self.trainer.init_model = self.trainer.init_model.eval()
            self.trainer.init_model.set_evaluation(True)
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
            'mse':0,
            'kl_penalty':0,
            'reward_model_score':0
        }
        num_batches = len(self.test_ds['pre'])
        with torch.no_grad():
            for i, batch in enumerate(self.test_ds['pre']):
                
                #Preprocess batch
                batch = preprocess_batch(batch, gpu_id=self.gpu_id, return_c=True)
                
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
                    val_metrics['reward'] += metrics['reward']
                    val_metrics['kl_penalty'] += metrics['kl_penalty']
                    val_metrics['reward_model_score'] += metrics['reward_model_score']

                except Exception as e:
                    print(traceback.format_exc())
                    continue
        
        loss = val_metrics['mse']/num_batches
        kl = val_metrics['kl_penalty']/num_batches
        reward_model_score = val_metrics['reward_model_score']/num_batches
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
            "val_ovl_reward":reward,
            "val_KL":kl,
            "reward_model_score":reward_model_score
        }) 
        print(f"Episode:{episode} | VAL_PESQ:{np.asarray(val_metrics['pesq']).mean()} | VAL_LOSS:{loss} | RM_SCORE: {reward_model_score}")
        
        self.actor.train()
        if self.args.method == 'PPO':
            self.actor.set_evaluation(False)
            #self.critic.train()
        
        return loss, reward_model_score, np.asarray(val_metrics["pesq"]).mean()

    def train_one_epoch(self, epoch):       
        #Run training
        self.actor.train()
        #if self.args.method == 'PPO':
            #self.critic.train()

        #loss, best_rm_score, best_pesq = self.run_validation(0)
        loss, best_rm_score, best_pesq = 9999, 0, 0
        epochs_per_episode = self.args.ep_per_episode
        
        run_validation_step = 250 // (epochs_per_episode * self.args.episode_steps)
        print(f"Run validation at every step:{run_validation_step}")
        best_val_loss = 99999
        episode_per_epoch = 100

        for i in range(episode_per_epoch):
            try:
                if self.args.method == 'reinforce': 
                    loss, kl, reward, pesq = self.trainer.run_episode(self.actor, self.optimizer)

                    wandb.log({
                        "episode": (i+1) + ((epoch - 1) * episode_per_epoch),
                        "Return":reward[1].item(),
                        "RM_Score":reward[0].item(),
                        "pg_loss":loss[0],
                        "pretrain_loss":loss[1],
                        "train_PESQ":pesq,
                        "Episode_KL":kl,
                    })
                    print(f"Epoch:{epoch} | Episode:{i+1} | Return: {reward[1]} | RM_Score: {reward[0]} | KL: {kl} | PESQ: {pesq}")

                if self.args.method == 'PPO':
                    loss, batch_reward, pesq = self.trainer.run_episode(self.actor, (self.optimizer, self.c_optimizer), n_epochs=epochs_per_episode)
                        
                    if loss is not None:
                        wandb.log({
                            "episode": (i+1) + ((epoch - 1) * episode_per_epoch),
                            "episode_avg_kl":batch_reward[1].item(),
                            "cumulative_G_t": batch_reward[0].item(),
                            #"critic_values": batch_reward[1].item(), 
                            "episodic_avg_r": batch_reward[3].item(),
                            "episodic_reward_model_score": batch_reward[2].item(),
                            #"advantages":adv,
                            "clip_loss":loss[0],
                            #"value_loss":loss[1],
                            "pretrain_loss":loss[3],
                            "pg_loss":loss[2],
                            "entropy_loss":loss[1],
                            "train_pesq":pesq, 
                        })

                        print(f"Epoch:{epoch} | Episode:{i+1} | Return: {batch_reward[0].item()} | Values: {batch_reward[1].item()}")
                
                if (i+1) % 10 == 0:
                #Run validation after each episode
                    loss, rm_score, val_pesq = self.run_validation((epoch-1) * episode_per_epoch + (i+1))
                    if val_pesq >= best_pesq or rm_score >= best_rm_score:
                        best_pesq = val_pesq
                        best_rm_score = max(best_rm_score, rm_score)
                        #self.trainer.init_model = copy.deepcopy(self.actor)
                        #self.trainer.init_model = self.trainer.init_model.eval()
                        self.save(loss, rm_score, val_pesq, (epoch-1) * episode_per_epoch + (i+1))
                    
                    #elif rm_score > best_rm_score:
                    #    best_rm_score = rm_score
                    #    self.save(loss, rm_score, val_pesq, (epoch-1) * episode_per_epoch + (i+1))

                    
                        
            except Exception as e:
                print(traceback.format_exc())
                continue

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
            if self.args.method == 'reinforce':
                save_dict = {'actor_state_dict':self.actor.state_dict(), 
                            'optim_state_dict':self.optimizer.state_dict()
                            }
            if self.args.method == 'PPO':
                save_dict = {'actor_state_dict':self.actor.state_dict(), 
                            'critic_state_dict':self.critic.state_dict(),
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




