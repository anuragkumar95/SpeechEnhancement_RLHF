import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
import torch
import torch.nn as nn
from utils import LearnableSigmoid


class QNet(nn.Module):
    def __init__(self, ndf, in_channel=2, gpu_id=None):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, ndf, (3, 3), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (3, 3), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (3, 3), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (3, 3), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),
        )
        self.gpu_id = gpu_id

    def forward(self, x, y, t):
        x1 = x['est_mag']
        x2 = x['clean_mag']
        
        m_mask = y[0]
        m_mask = m_mask.permute(0, 1, 3, 2).squeeze(-1)
        mask = torch.ones(x1.shape)
        if self.gpu_id is not None:
            mask = mask.to(self.gpu_id)
        mask[:, :, :, t] = m_mask
        
        x1 = x1*mask
        xy = torch.cat([x1, x2], dim = 1)
        
        return self.layers(xy)