"""
Local Attention as describe in Stand Alone Self-Attention
https://arxiv.org/abs/1906.05909
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import math

from models.transformer_encoder import Encoder
from models.modules.local_attention import LocalAttention

class LAT(nn.Module):
    """
    Local Attention Encoder
    """
    def __init__(self, feature_size, no_of_blocks, max_pooling_blocks=3, dropout = 0., memory_block_size=4, kernel_size=2):
        super(LAT, self).__init__()
        # TODO Assert if memory block sizr, kernel size, noOf block AND feature size are compatible

        encoders = []
        in_size = feature_size
        for i in range(no_of_blocks - max_pooling_blocks):
            attention = LocalAttention(in_size, in_size, memory_block_size, 1)
            encoders.append(Encoder(in_size, 1, in_size, attention, dropout = dropout, with_avg_pooling=False))

        for i in range(max_pooling_blocks):
            attention = LocalAttention(in_size, in_size, memory_block_size, kernel_size)
            out_size = feature_size // kernel_size ** (2 * (i + 1))
            encoders.append(Encoder(out_size, 1, out_size, attention, dropout = dropout, with_avg_pooling=True))
            in_size = out_size

        self.transformer = nn.Sequential(*encoders)

    def forward(self, x):
        """
        Takes input image as a tensor of shape (batch_size, channel, height * width)
        """
        return self.transformer(x)
