import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class Maxout(nn.Module):
    def __init__(self, k):
        super(Maxout, self).__init__()
        self.k = k

    def forward(self, x):
        # assert x.shape
        torch.max(x)


class Generator(nn.Module):
    def __init__(self, z_dim=8000, hidden_dim=3072):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        layers = [
            ('dense0', nn.Linear(self.z_dim, hidden_dim)),
            ('dense0_act', nn.Sigmoid()),
            ('dense1', nn.Linear(self.hidden_dim, hidden_dim))
        ]

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=3072):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        layers = [
            ('dense0', nn.Linear(self.input_dim, hidden_dim)),
            ('dense0_act', nn.Sigmoid()),
            ('dense1', nn.Linear(self.hidden_dim, hidden_dim))
        ]

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x)
