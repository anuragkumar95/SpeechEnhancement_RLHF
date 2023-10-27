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

    def forward(self, x, y, t):
        next_state = self.agent.get_next_state(x, y, t)
        """
        x_real = x['est_real']
        x_imag = x['est_imag']
        mag = x['est_mag']
        mask = y[0].permute(0, 1, 3, 2)
        complex_out = y[1]
        
        noisy_phase = torch.angle(torch.complex(x_real, x_imag))
        out_mag = mask * mag[:, :, :, t].unsqueeze(-1)

        mag_real = out_mag * torch.cos(noisy_phase[:, :, :, t].unsqueeze(-1))#.permute(0, 1, 3, 2)
        mag_imag = out_mag * torch.sin(noisy_phase[:, :, :, t].unsqueeze(-1))#.permute(0, 1, 3, 2)

        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)
        est_spec_uncompress = power_uncompress(final_real, final_imag).squeeze(1).permute(0,3,1,2)

        final_mag = torch.sqrt(est_spec_uncompress[:, 0, :, :]**2 + est_spec_uncompress[:, 1, :, :]**2)
        final_mag = final_mag.squeeze(-1).unsqueeze(1)
        mag[:, :, :, t] = final_mag
        clean_mag = x['clean_mag']
        """
        mag = next_state['est_mag']
        clean_mag = next_state['clean_mag']

        xy = torch.cat([mag, clean_mag], dim = 1)
        return self.layers(xy)