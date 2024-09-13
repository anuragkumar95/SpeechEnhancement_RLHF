from model.CMGAN.conformer import ConformerBlock
import torch
import torch.nn as nn
#from model.critic import QNet
#import torch.nn.functional as F

from torch.distributions import Normal
#from torch.nn.utils import rnn


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class TSCB(nn.Module):
    def __init__(self, num_channel=64, out_channel=2, nheads=4):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=nheads,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.freq_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=nheads,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out



class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1, gpu_id=None, eval=False):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)
        self.gpu_id = gpu_id
        self.evaluation = eval

    def sample(self, mu, x=None):
        sigma = (torch.ones(mu.shape)*0.01).to(self.gpu_id) 
        N = Normal(mu, sigma)
        if x is None:
            x = N.rsample()
        if x.shape != mu.shape:
            raise ValueError(f"Dims in action {x.shape} don't match mu {mu.shape}")
        x_logprob = N.log_prob(x)
        x_entropy = N.entropy()
        return x, x_logprob, x_entropy, (mu, sigma)

    def forward(self, x, action=None):
       
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        x_mu = self.prelu_out(x)
        x, x_logprob, x_entropy, params = self.sample(x_mu, action)
        #print(f"Mask dec: X_MU:{x_mu.shape}")
        if self.evaluation:
            x = x_mu
        return x, x_logprob, x_entropy, params

class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64, gpu_id=None, eval=False):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))
        self.gpu_id = gpu_id
        self.evaluation = eval
       
    def sample(self, mu, x=None):
        sigma = (torch.ones(mu.shape) * 0.01).to(self.gpu_id) 
        N = Normal(mu, sigma)
        if x is None:
            x = N.rsample()
        if x.shape != mu.shape:
            raise ValueError(f"Dims in action {x.shape} don't match mu {mu.shape}")
        
        x_logprob = N.log_prob(x)
        x_entropy = N.entropy()
        return x, x_logprob, x_entropy, (mu, sigma)

    def forward(self, x, action=None):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x_mu = self.conv(x)
        x, x_logprob, x_entropy, params = self.sample(x_mu, action) 
        if self.evaluation:
            x = x_mu
        return x, x_logprob, x_entropy, params
        

class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201, gpu_id=None, eval=False):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel, nheads=4)
        self.TSCB_2 = TSCB(num_channel=num_channel, nheads=4)
        self.TSCB_3 = TSCB(num_channel=num_channel, nheads=4)
        self.TSCB_4 = TSCB(num_channel=num_channel, nheads=4)
        
        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1, gpu_id=gpu_id, eval=eval
        )
        self.complex_decoder = ComplexDecoder(num_channel=num_channel, gpu_id=gpu_id, eval=eval)
        self.gpu_id = gpu_id

    def set_evaluation(self, bool):
        self.mask_decoder.evaluation = bool
        self.complex_decoder.evaluation = bool

    def get_action(self, x):
        #b, ch, t, f = x.size()
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        mask, m_logprob, m_entropy, params = self.mask_decoder(out_5)
        complex_out, c_logprob, c_entropy, c_params = self.complex_decoder(out_5)

        print(f"get_Action: mask:{mask.shape}, comp:{complex_out.shape}")

        return (mask, complex_out), (m_logprob, c_logprob), (m_entropy, c_entropy), (params, c_params)

    
    def get_action_prob(self, x, action=None):
        """
        ARGS:
            x : spectrogram
            action : (Tuple) Tuple of mag and complex actions

        Returns:
            Tuple of mag and complex masks log probabilities.
        """
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)
      
        _, m_logprob, m_entropy, _ = self.mask_decoder(out_5, action[0])
        _, c_logprob, c_entropy, _ = self.complex_decoder(out_5, action[1])

        return (m_logprob, c_logprob), (m_entropy, c_entropy)
        
        
    def get_embedding(self, x):
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        
        x_in = torch.cat([mag, x], dim=1)
        
        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        return out_5

    def forward(self, x):
        #b, ch, t, f = x.size() 
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
        
        x_in = torch.cat([mag, x], dim=1)
        
        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        mask, _, _, _ = self.mask_decoder(out_5)
        complex_out, _, _, _ = self.complex_decoder(out_5)
        
        mask = mask.permute(0, 2, 1).unsqueeze(1)
        out_mag = mask * mag
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag
 