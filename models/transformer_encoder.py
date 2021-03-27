"""
Transformer Encoder
"""

# TODO Reimplement

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math


class Residual(nn.Module):
    def __init__(self, fn, with_avg_pooling=False, kernel_size=2):
        super().__init__()
        self.fn = fn
        self.with_avg_pooling = with_avg_pooling
        if with_avg_pooling:
            self.avg_pool = nn.AvgPool2d(kernel_size)
    def forward(self, x, **kwargs):
        if self.with_avg_pooling:
            #TODO clean that up please sir
            height = int(math.sqrt(x.shape[2]))
            x = self.avg_pool((self.fn(x, **kwargs) + x).view(-1, x.shape[1], height, height))
            return x.view(-1, x.shape[1], x.shape[2] * x.shape[2])
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, with_avg_pooling=False, kernel_size=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim if not with_avg_pooling else dim * kernel_size ** 2)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, dim, no_of_blocks, mlp_dim, attention, dropout = 0., with_avg_pooling=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(no_of_blocks):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, attention, with_avg_pooling=with_avg_pooling), with_avg_pooling=with_avg_pooling),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x
