import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, state, action):
        next_state = self.agent.get_next_state(state, action)
        
        mag = next_state['est_mag']
        clean_mag = next_state['clean_mag']

        xy = torch.cat([mag, clean_mag], dim = 1)
        #yy = torch.cat([clean_mag, clean_mag], dim = 1)
        #return 1 - (self.layers(yy) - self.layers(xy))
        #[0.1, 0.9, 0.2, 0.05]
        loss_mag = F.mse_loss(next_state['clean_mag'], next_state['est_mag'])
        loss_real = F.mse_loss(next_state['clean_real'],next_state['est_real'])
        #time_loss = F.mse_loss(next_state['cl_audio'], next_state['est_audio'])

        loss = 0.1 * loss_real + 0.9 * loss_mag# + 0.2 * time_loss

        return self.layers(xy) + (1 - loss.mean()) 