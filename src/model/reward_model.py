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
        self.eps = 1e-08
        self.loss = torch.nn.CrossEntropyLoss(reduce='mean')
        
    def forward(self, pos, neg, labels):

        x_pos = pos.permute(0, 1, 3, 2)
        x_neg = neg.permute(0, 1, 3, 2)

        pos_emb = self.conformer.get_embedding(x_pos)
        neg_emb = self.conformer.get_embedding(x_neg)
        
        pos_proj = F.sigmoid(self.reward_projection(pos_emb))
        neg_proj = F.sigmoid(self.reward_projection(neg_emb))

        #score = F.sigmoid(pos_proj - neg_proj)
        score = torch.cat([pos_proj, neg_proj], dim=-1)
        probs = F.softmax(score)
        #loss = -torch.log(score + self.eps)
        loss = self.loss(score, labels) 
   
        return loss, score, probs
    
    def get_reward(self, inp):
        """
        ARGS:
            inp : spectrogram of curr state (b * ch * t * f) 

        Returns
            Reward in the range (0, 1) for next state with reference to curr state.
        """
        inp = inp.permute(0, 1, 3, 2)
        
        inp_emb = self.conformer.get_embedding(inp)
       
        proj = self.reward_projection(inp_emb)

        #Restrict rewards to be in range (0, 1)
        rewards = F.sigmoid(proj)

        return rewards
