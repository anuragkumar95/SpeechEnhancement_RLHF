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
        self.reward_projection = QNet(ndf=16, in_channel=64, out_channel=1)
        
    def forward(self, x_ref, x_per):

        x_ref = x_ref.permute(0, 1, 3, 2)
        x_per = x_per.permute(0, 1, 3, 2)

        ref_emb = self.conformer.get_embedding(x_ref)
        per_emb = self.conformer.get_embedding(x_per)

        print(f"ref:{ref_emb.shape}, per:{per_emb.shape}")
        
        score_ref = self.reward_projection(ref_emb)
        score_per = self.reward_projection(per_emb)

        scores = torch.cat([score_ref, score_per], dim=-1)

        print(f"proj:{scores.shape}")
        probs = F.softmax(scores, dim=-1)
        print(f"probs:{probs.shape}")
        print(f"PROBS:{probs}")
        return probs
    
    def get_reward(self, x):
        """
        ARGS:
            x : spectrogram of shape (b * ch * t * f)
        """
        x = x.permute(0, 1, 3, 2)
        x_emb = self.conformer.get_embedding(x)
        
        rewards = self.reward_projection(x_emb)

        return rewards
