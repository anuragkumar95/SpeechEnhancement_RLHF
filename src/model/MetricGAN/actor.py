

#Taken from https://github.com/wooseok-shin/MetricGAN-OKD/blob/main/SE/model.py

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torch.distributions import Normal


def xavier_init_layer(
    in_size, out_size=None, spec_norm=True, layer_type=nn.Linear, **kwargs
):
    "Create a layer with spectral norm, xavier uniform init and zero bias"
    if out_size is None:
        out_size = in_size

    layer = layer_type(in_size, out_size, **kwargs)
    if spec_norm:
        layer = spectral_norm(layer)

    # Perform initialization
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)

    return layer


class Learnable_sigmoid(nn.Module):
    def __init__(self, in_features=257):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True  # set requiresGrad to true!

    def forward(self, x):
        return 1.2 * torch.sigmoid(self.slope * x)


class Generator(nn.Module):
    def __init__(self, causal=False, gpu_id=None, eval=False):
        super(Generator, self).__init__()
        dim = 200
        self.lstm = nn.LSTM(257, dim, dropout=0, num_layers=2, bidirectional=not causal, batch_first=True)    # causal==False -> bidirectional=True
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0
        """
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        self.LReLU = nn.LeakyReLU(negative_slope=0.3)
        if not causal:
            dim = dim * 2
        self.fc1 = xavier_init_layer(dim, 300, spec_norm=False)
        self.fc2 = xavier_init_layer(300, 257, spec_norm=False)
        self.evaluation = eval
        self.gpu_id = gpu_id
        
        self.Learnable_sigmoid = Learnable_sigmoid()

    def set_evaluation(self, bool):
        self.evaluation = bool

    def sample(self, mu, x=None):
        sigma = (torch.ones(mu.shape) * 0.01).to(self.gpu_id) 
        N = Normal(mu, sigma)
        print(f"normal:{mu.mean(), sigma.mean()}")
        if x is None:
            x = N.rsample()
        x_logprob = N.log_prob(x)
        x_entropy = N.entropy()
        return x, x_logprob, x_entropy, (mu, sigma)
        
    def forward(self, x, lengths=None, action=None):
        # Pack sequence for LSTM padding
        mag = x

        if lengths is not None:
            mag = self.pack_padded_sequence(mag, lengths)
        print(f"FORWARD: inp:{mag.mean()}")
        outputs, _ = self.lstm(mag)
        print(f"FORWARD: lstm:{outputs.mean()}")
        # Unpack the packed sequence
        if lengths is not None:
            outputs = self.pad_packed_sequence(outputs)

        outputs = self.fc1(outputs)
        print(f"FORWARD: fc1:{outputs.mean()}")
        outputs = self.LReLU(outputs)
        #outputs = self.fc2(outputs)
        #outputs = self.Learnable_sigmoid(outputs)
        print(f"FORWARD: fc2 inp :{outputs.max(), outputs.min(), outputs.mean()}")
        x = self.fc2(outputs)
        print(f"FORWARD: fc2 out :{x.max(), x.min(), x.mean()}")
        x_mu = self.Learnable_sigmoid(x)
        x, x_logprob, x_entropy, params = self.sample(x_mu, action)
        if self.evaluation:
            x = x_mu
        return x, x_logprob, x_entropy, params
    
    def get_action_prob(self, x, action):
        """
        ARGS:
            x : spectrogram
            action : (Tuple) Tuple of mag and complex actions

        Returns:
            Tuple of mag and complex masks log probabilities.
        """
        _, x_logprob, x_entropy, _ = self.forward(x, None, action)
        return x_logprob, x_entropy
    
    def get_action(self, x):
        action, x_logprob, x_entropy, params = self.forward(x, None, None)   
        return action, x_logprob, x_entropy, params

    def pack_padded_sequence(self, inputs, lengths):
        lengths = lengths.cpu()
        return torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)

    def pad_packed_sequence(self, inputs):
        outputs, lengths = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, num_target_metrics=1):
        super(Discriminator, self).__init__()

        self.BN = nn.BatchNorm2d(num_features=2, momentum=0.01)

        layers = []
        base_channel = 16
        layers.append(xavier_init_layer(2, base_channel, layer_type=nn.Conv2d, kernel_size=(5,5)))
        layers.append(xavier_init_layer(base_channel, base_channel*2, layer_type=nn.Conv2d, kernel_size=(5,5)))
        layers.append(xavier_init_layer(base_channel*2, base_channel*4, layer_type=nn.Conv2d, kernel_size=(5,5)))
        layers.append(xavier_init_layer(base_channel*4, base_channel*8, layer_type=nn.Conv2d, kernel_size=(5,5)))
        self.layers = nn.ModuleList(layers)
        
        self.LReLU = nn.LeakyReLU(0.3)
        
        self.fc1 = xavier_init_layer(base_channel*8, 50)
        self.fc2 = xavier_init_layer(50, 10)
        self.fc3 = xavier_init_layer(10, num_target_metrics)

    def forward(self, x):
        x = self.BN(x)
        for layer in self.layers:
            x = layer(x)
            x = self.LReLU(x)

        x = torch.mean(x, (2, 3))    # Average Pooling
        x = self.LReLU(self.fc1(x))
        x = self.LReLU(self.fc2(x))

        x = self.fc3(x)
        return x
    
    