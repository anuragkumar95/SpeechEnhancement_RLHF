from model.conformer import ConformerBlock
import torch
import torch.nn as nn
from model.critic import QNet
import torch.nn.functional as F

from torch.distributions import Normal, Categorical
from torch.nn.utils import rnn


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
    def __init__(self, num_features, num_channel=64, out_channel=1, distribution=None, gpu_id=None, eval=False):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        if distribution == "Normal":
            self.final_conv_mu = nn.Conv2d(out_channel, out_channel, (1, 1))
            self.final_conv_var = nn.Conv2d(out_channel, out_channel, (1, 1))
        else:
            self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)
        self.gpu_id = gpu_id
        self.dist = distribution
        self.evaluation = eval

    def sample(self, mu, logvar, x=None):
        if self.dist == 'Normal':
            sigma = torch.clamp(torch.exp(logvar) + 1e-08, min=1.0)
        elif self.dist is None:
            sigma = (torch.ones(mu.shape)*0.01).to(self.gpu_id) 
        N = Normal(mu, sigma)
        if x is None:
            x = N.rsample()
        x_logprob = N.log_prob(x)
        x_entropy = N.entropy()
        return x, x_logprob, x_entropy, (mu, sigma)

    def forward(self, x, action=None):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        if self.dist == 'Normal':
            x_mu = self.final_conv_mu(x).permute(0, 3, 2, 1).squeeze(-1)
            x_var = self.final_conv_var(x).permute(0, 3, 2, 1).squeeze(-1)
            x, x_logprob, x_entropy, params = self.sample(x_mu, x_var, action)
            x_out = self.prelu_out(x)
            
        elif self.dist is None:
            x_mu = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
            x, x_logprob, x_entropy, params = self.sample(x_mu, None, action)
            if self.eval:
                x_out = self.prelu_out(params[0])
            else:
                x_out = self.prelu_out(x)
        return (x, x_out), x_logprob, x_entropy, params

class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64, distribution=None, gpu_id=None, eval=False):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        if distribution=='Normal':
            self.conv_mu = nn.Conv2d(num_channel, 2, (1, 2))
            self.conv_var = nn.Conv2d(num_channel, 2, (1, 2))
        else:
            self.conv = nn.Conv2d(num_channel, 2, (1, 2))
        self.out_dist = distribution
        self.gpu_id = gpu_id
        self.evaluation = eval
       
    def sample(self, mu, logvar, x=None):
        if self.out_dist == 'Normal':
            sigma = torch.clamp(torch.exp(logvar) + 1e-08, min=0.01)
        elif self.out_dist is None:
            sigma = (torch.ones(mu.shape) * 0.01).to(self.gpu_id) 
        N = Normal(mu, sigma)
        if x is None:
            x = N.rsample()
        x_logprob = N.log_prob(x)
        x_entropy = N.entropy()
        return x, x_logprob, x_entropy, (mu, sigma)

    def forward(self, x, action=None):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))

        if self.out_dist == "Normal":
            x_mu = self.conv_mu(x)
            x_var = self.conv_var(x)
            x, x_logprob, x_entropy, params = self.sample(x_mu, x_var, action)
            #return x, x_logprob, x_entropy, params
        
        if self.out_dist is None:
            x_mu = self.conv(x)
            x, x_logprob, x_entropy, params = self.sample(x_mu, None, action)
            if self.eval:
                x = params[0]
        return x, x_logprob, x_entropy, params
        
class TSCNetSmall(nn.Module):
    def __init__(self, num_channel=64, num_features=201, distribution=None, gpu_id=None, eval=False):
        super(TSCNetSmall, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel, nheads=4)
        
        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1, distribution=distribution, gpu_id=gpu_id, eval=eval,
        )
        self.complex_decoder = ComplexDecoder(num_channel=num_channel, distribution=distribution, gpu_id=gpu_id, eval=eval)
        self.dist = distribution

    def set_evaluation(self, bool):
        self.mask_decoder.evaluation = bool
        self.complex_decoder.evaluation = bool

    def get_action(self, x):
        #b, ch, t, f = x.size()
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)

        #if self.dist=='Normal':
        mask, m_logprob, m_entropy, params = self.mask_decoder(out_2)
        complex_out, c_logprob, c_entropy, c_params = self.complex_decoder(out_2)
        return (mask, complex_out), (m_logprob, c_logprob), (m_entropy, c_entropy), (params, c_params)
        
        #mask = self.mask_decoder(out_2)
        #complex_out = self.complex_decoder(out_2)
        #return (mask, complex_out), None, None
    
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

        _, m_logprob, m_entropy, _ = self.mask_decoder(out_2, action[0][0])
        _, c_logprob, c_entropy, _ = self.complex_decoder(out_2, action[1])

        return (m_logprob, c_logprob), (m_entropy, c_entropy)
        
        
    def get_embedding(self, x):
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        
        x_in = torch.cat([mag, x], dim=1)
        
        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)

        return out_2

    def forward(self, x):
        b, ch, t, f = x.size() 
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
        
        x_in = torch.cat([mag, x], dim=1)
        
        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)

        #if self.dist == "Normal":
        (_, mask), _, _, m_params = self.mask_decoder(out_2)
        complex_out, _, _, c_params = self.complex_decoder(out_2)
        
        #if self.dist == "None":
        #    mask = self.mask_decoder(out_2)
        #    complex_out = self.complex_decoder(out_2)
        
        mask = mask.permute(0, 2, 1).unsqueeze(1)
        out_mag = mask * mag
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag
        

class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201, distribution=None, gpu_id=None, eval=False):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel, nheads=4)
        self.TSCB_2 = TSCB(num_channel=num_channel, nheads=4)
        self.TSCB_3 = TSCB(num_channel=num_channel, nheads=4)
        self.TSCB_4 = TSCB(num_channel=num_channel, nheads=4)
        
        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1, distribution=distribution, gpu_id=gpu_id, eval=eval
        )
        self.complex_decoder = ComplexDecoder(num_channel=num_channel, distribution=distribution, gpu_id=gpu_id, eval=eval)
        self.dist = distribution

    def set_evaluation(self, bool):
        self.mask_decoder.evaluation = bool
        self.complex_decoder.evaluation = bool

    def get_action(self, x):
        b, ch, t, f = x.size()
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        mask, m_logprob, m_entropy, params = self.mask_decoder(out_5)
        complex_out, c_logprob, c_entropy, c_params = self.complex_decoder(out_5)
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
        
        _, m_logprob, m_entropy, _ = self.mask_decoder(out_5, action[0][0])
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

        (_, mask), _, _, m_params = self.mask_decoder(out_5)
        complex_out, _, _, c_params = self.complex_decoder(out_5)
        
        mask = mask.permute(0, 2, 1).unsqueeze(1)
        out_mag = mask * mag
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag
 
    

class LSTMActor(nn.Module):
    def __init__(self,
                 generator, 
                 n_layers, 
                 hid_dim,
                 in_channel, 
                 in_dim, 
                 out_dim, 
                 batch_first=True,
                 drop_out=0.05,
                 bi_directional=False,
                 gpu_id=None):
        
        self.encoder = generator

        self.gru = nn.GRU(input_size=in_dim * in_channel,
                          hid_dim=hid_dim,
                          num_layers=n_layers,
                          batch_first=batch_first,
                          drop_out=drop_out,
                          bidirectional=bi_directional)
        
        self.num_layers = n_layers
        self.hidden_dim = hid_dim
        self.bi_directional = bi_directional
        
        #FC layers to output a dist over K classes
        inp_dim = hid_dim
        if self.bi_directional:
            inp_dim = 2 * inp_dim
        self.final_out = nn.Linear(inp_dim, out_dim)

        self.gpu_id = gpu_id

    def init_hidden_state(self, h):
        return nn.init.xavier_uniform_(h, gain=nn.init.calculate_gain('relu'))

    def forward(self, inputs, lens, batch_first=False):
        """
        ARGS:
            inputs : (Long.tensor) padded tensor of shape (batch,ch,t, f).
            lens   : (List[Int]) length of time_seq for each example in the batch. 
        """  
        # Initializing hidden state for first input with zeros
        num_layers = self.num_layers
        if self.bi_directional:
            num_layers = 2*self.num_layers
        if batch_first:
            h0 = torch.zeros(num_layers, inputs.shape[0], self.hidden_dim).requires_grad_()
        else:
            h0 = torch.zeros(inputs.shape[0], num_layers, self.hidden_dim).requires_grad_()
        h0 = self.init_hidden_state(h0)

        if self.gpu_id is not None:
            h0 = h0.to(self.gpu_id)
        
        #get embeddings from the generator
        encodings = self.encoder.get_embeddings(inputs)
        #change (b, ch, t, f) --> (b, t, f * ch)
        b, c, t, f = encodings.size()
        encodings = encodings.permute(0, 2, 3, 1).contiguous().view(b, t, f*c)

        #Feed-forward GRU
        packed_inp = rnn.pack_padded_sequence(encodings, lens, batch_first=batch_first, enforce_sorted=False)
        gru_outputs, _ = self.gru(packed_inp, h0)
        gru_outputs, lens = rnn.pad_packed_sequence(packed_inp, batch_first=batch_first)

        #Predict probs over K clusters
        scores = self.final_out(gru_outputs)
        probs = F.Softmax(scores, dim=-1)

        return probs
        



