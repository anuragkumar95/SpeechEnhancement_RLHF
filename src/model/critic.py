import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
import torch
import torch.nn as nn
from utils import LearnableSigmoid


class QNet(nn.Module):
    def __init__(self, ndf, in_channel=2):
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

    def forward(self, x, y, t):
        x1 = x['est_mag'][:, :, :, t].unsqueeze(-1)
        x2 = x['est_real'][:, :, :, t].unsqueeze(-1)
        x3 = x['est_imag'][:, :, :, t].unsqueeze(-1)
        m_mask, c_out = y
        m_mask = m_mask.permute(0, 1, 3, 2)

        x1 = torch.cat([x1, m_mask], dim=-1)
        x2 = torch.cat([x2, c_out[:, 0, :, :].unsqueeze(1)], dim=-1)
        x3 = torch.cat([x3, c_out[:, 1, :, :].unsqueeze(1)], dim=-1)

        x = torch.cat([x1, x2, x3], dim = 1)
        #xy = torch.cat([x, y], dim=1)
        return self.layers(x)