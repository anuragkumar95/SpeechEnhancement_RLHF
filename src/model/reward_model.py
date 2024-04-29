# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import torch
import torch.nn as nn
from model.critic import QNet
import torch.nn.functional as F



class RewardModel(nn.Module):
    def __init__(self, policy, in_channels=2):
        super(RewardModel, self).__init__()
        self.encoder = policy 
        self.reward_projection = QNet(ndf=16, in_channel=in_channels, out_channel=1)
        self.eps = 1e-08
   
        
    def forward(self, x, pos, neg):
        """
        x is (b * ch * f * t)
        """
        x = x.permute(0, 3, 1, 2)
        x_pos = pos.permute(0, 3, 1, 2)
        x_neg = neg.permute(0, 3, 1, 2)


        x_encoded = self.encoder.get_embedding(x)
        pos_encoded = self.encoder.get_embedding(x_pos)
        neg_encoded = self.encoder.get_embedding(x_neg)

        print(f"x:{x_encoded.shape}, pos:{pos_encoded.shape}, neg:{neg_encoded.shape}")
        
        pos_inp = torch.cat([x_encoded, pos_encoded], dim=1)
        neg_inp = torch.cat([x_encoded, neg_encoded], dim=1)

        print(f"pos_inp:{pos_inp.shape}, neg_inp:{neg_inp.shape}")

        pos_proj = self.reward_projection(pos_inp)
        neg_proj = self.reward_projection(neg_inp)

        loss = -torch.log(F.sigmoid(pos_proj - neg_proj) + self.eps).mean()
   
        return loss, (pos_proj, neg_proj), None
    
    def get_reward(self, inp, out):
        """
        ARGS:
            inp : spectrogram of curr state (b * ch * t * f) 

        Returns
            Reward in the range (0, 1) for next state with reference to curr state.
        """
        inp = inp.permute(0, 1, 3, 2)
        out = out.permute(0, 1, 3, 2)

        proj_inp = self.reward_projection(torch.cat([inp, out], dim=1))
        
        return proj_inp
