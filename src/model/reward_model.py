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
        self.eps = 1e-08
   
        
    def forward(self, pos, neg):

        x_pos = pos.permute(0, 1, 3, 2)
        x_neg = neg.permute(0, 1, 3, 2)

        pos_proj = F.sigmoid(self.reward_projection(x_pos))
        neg_proj = F.sigmoid(self.reward_projection(x_neg))


        score_proj = F.sigmoid(pos_proj - neg_proj) 

        loss = -torch.log(score_proj + self.eps).mean()
   
        return loss, score_proj
    
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
