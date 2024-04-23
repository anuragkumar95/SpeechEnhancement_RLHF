# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import torch
import torch.nn as nn
from model.critic import QNet
import torch.nn.functional as F



class RewardModel(nn.Module):
    def __init__(self, in_channels=2):
        super(RewardModel, self).__init__()
        self.reward_projection = QNet(ndf=16, in_channel=in_channels, out_channel=1)
        self.eps = 0.2
   
        
    def forward(self, pos, neg, label):

        x_pos = pos.permute(0, 1, 3, 2)
        x_neg = neg.permute(0, 1, 3, 2)

        pos_proj = self.reward_projection(x_pos)
        neg_proj = self.reward_projection(x_neg)

        #loss = -torch.log(F.sigmoid(pos_proj - neg_proj) - self.eps).mean()

        dist = pos_proj - neg_proj

        #label is one_hot_vector
        label = torch.argmax(label, dim=-1)
        #loss = (label * (pos_proj - neg_proj)**2 + (1 - label) * (torch.clamp(self.eps - (pos_proj - neg_proj), min=0))**2 ).mean()

        loss1 = (label) * torch.pow(dist, 2)
        loss2 = (1 - label) * torch.pow(torch.clamp(self.eps - dist, min=0.0), 2)
        loss = (loss1 + loss2).mean() 
        
        print(f"loss_1:{loss1.mean()}, loss_2:{loss2.mean()}")
        
   
        return loss, (pos_proj, neg_proj), None
    
    def get_reward(self, inp):
        """
        ARGS:
            inp : spectrogram of curr state (b * ch * t * f) 

        Returns
            Reward in the range (0, 1) for next state with reference to curr state.
        """
        inp = inp.permute(0, 1, 3, 2)
        #out = out.permute(0, 1, 3, 2)

        #x = torch.cat([inp, out], dim=1)
       
        proj_inp = self.reward_projection(inp)
        #proj_out = self.reward_projection(out)

        return proj_inp
