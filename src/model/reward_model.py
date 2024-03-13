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
        
    def forward(self, anchor, pos, neg):

        anchor = anchor.permute(0, 1, 3, 2)
        x_pos = pos.permute(0, 1, 3, 2)
        x_neg = neg.permute(0, 1, 3, 2)

        anchor_emb = self.conformer.get_embedding(anchor)
        pos_emb = self.conformer.get_embedding(x_pos)
        neg_emb = self.conformer.get_embedding(x_neg)

        pos_inp = torch.cat([anchor_emb, pos_emb], dim=1)
        neg_inp = torch.cat([anchor_emb, neg_emb], dim=1)

        #print(f"ref:{ref_inp.shape}, per:{per_inp.shape}")
        
        pos_proj = self.reward_projection(pos_inp)
        neg_proj = self.reward_projection(neg_inp)

        print(f"diff:{(pos_proj - neg_proj).mean(-1)}")

        score = F.sigmoid(pos_proj - neg_proj + self.eps)

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
