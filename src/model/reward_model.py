# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import torch
import torch.nn as nn
from model.critic import QNet
import torch.nn.functional as F



class RewardModel(nn.Module):
    def __init__(self, policy):
        super(RewardModel, self).__init__()
        self.conformer = policy
        self.reward_projection = QNet(ndf=16, in_channel=128, out_channel=1)
        self.eps = 1e-08
        
    def forward(self, x_ref, x_per):

        x_ref = x_ref.permute(0, 1, 3, 2)
        x_per = x_per.permute(0, 1, 3, 2)

        ref_emb = self.conformer.get_embedding(x_ref)
        per_emb = self.conformer.get_embedding(x_per)

        ref_inp = torch.cat([per_emb, ref_emb], dim=1)
        per_inp = torch.cat([per_emb, per_emb], dim=1)

        #print(f"ref:{ref_inp.shape}, per:{per_inp.shape}")
        
        ref_proj = self.reward_projection(ref_inp)
        per_proj = self.reward_projection(per_inp)

        print(f"diff:{torch.sqrt((ref_proj - per_proj)**2).mean(-1)}")

        score = F.sigmoid(torch.log( torch.sqrt((ref_proj - per_proj)**2) + self.eps))

        print(f"SCORE:{score.mean(-1)}")

        loss = -torch.log(score)
        print(f"LOSS:{loss.mean(-1)}")
        return loss.mean(), score
    
    def get_reward(self, x):
        """
        ARGS:
            x : spectrogram of shape (b * ch * t * f)
        """
        x = x.permute(0, 1, 3, 2)
        x_emb = self.conformer.get_embedding(x)
        
        rewards = self.reward_projection(x_emb)

        return rewards
