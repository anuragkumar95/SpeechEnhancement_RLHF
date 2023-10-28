import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
import torch
import torch.nn as nn
from utils import LearnableSigmoid, power_uncompress
from speech_enh_env import SpeechEnhancementAgent


class QNet(nn.Module):
    def __init__(self, ndf, in_channel=2, gpu_id=None):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)
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
        self.agent = SpeechEnhancementAgent(window=25, 
                                            buffer_size=1000,
                                            n_fft=400,
                                            hop=100,
                                            gpu_id=self.gpu_id,
                                            args=None)

    def forward(self, state, action, t):
        next_state = self.agent.get_next_state(state, action, t)
        
        mag = next_state['est_mag']
        clean_mag = next_state['clean_mag']

        xy = torch.cat([mag, clean_mag], dim = 1)
        yy = torch.cat([clean_mag, clean_mag], dim = 1)
        return 1 - (self.layers(yy) - self.layers(xy))