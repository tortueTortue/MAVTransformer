"""
Local Attention as describe in Stand Alone Self-Attention
https://arxiv.org/abs/1906.05909

Implemented by Leaderj1001

https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import math

from models.transformer_encoder import Encoder

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        # Switch for linears for performance
        self.key_conv = nn.Linear(in_channels, out_channels, bias=bias)
        self.query_conv = nn.Linear(in_channels, out_channels, bias=bias)
        self.value_conv = nn.Linear(in_channels, out_channels, bias=bias)
        # self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # self.key_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=bias)
        # self.query_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=bias)
        # self.value_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x, mask = None):
        # TODO FIX THIS CONV BS!!!
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
            
        if len(x.size()) == 3:
            height = width = int(math.sqrt(x.shape[2]))
            x = torch.reshape(x, (x.shape[0], x.shape[1], height, width))
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return torch.reshape(out, (out.shape[0], out.shape[1] * out.shape[2], out.shape[3]))

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class LAT(nn.Module):
    """
    Local Attention Encoder
    """
    def __init__(self, feature_size, no_of_blocks, mlp_dim, dropout = 0.):
        super(LAT, self).__init__()
        
        first_attention = AttentionConv(1, mlp_dim, kernel_size=7, padding=3, groups=8)
        self.first_encoder = Encoder(mlp_dim, no_of_blocks, feature_size, first_attention, dropout = dropout)

        attention = AttentionConv(mlp_dim, mlp_dim, kernel_size=7, padding=3, groups=8)
        self.encoder = Encoder(mlp_dim, no_of_blocks, mlp_dim, attention, dropout = dropout)

    def forward(self, x):
        return self.encoder(self.first_encoder(x))
