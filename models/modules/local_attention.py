import torch
from torch import nn
from einops.layers.torch import Rearrange

import math

class LocalAttention(nn.Module):
    def __init__(self, in_features, qvk_dim, memory_block_size, kernel_size):
        super(LocalAttention, self).__init__()

        output_height = int(math.sqrt(qvk_dim))
        assert qvk_dim % output_height == 0, "Feature map height and width has to be equal."
        assert output_height % memory_block_size == 0, "Feature map has to be divible by blocks."
        assert output_height % kernel_size == 0, "Feature map has to be divisible by kernel size."

        self.w_query = nn.Linear(in_features, qvk_dim, bias=False)
        self.w_key = nn.Linear(in_features, qvk_dim, bias=False)
        self.w_value = nn.Linear(in_features, qvk_dim, bias=False)

        self.mem_blocks_divider = Rearrange('b c (h m1) (w m2) -> b c (h w) (m1 m2)',
                                       m1=memory_block_size,m2=memory_block_size)

        self.soft = nn.Softmax(dim=4)

        self.avg_pool = nn.AvgPool2d(kernel_size)

    def forward(self, x, mask=None):
        """
        Here we assume that x is in the shape (b, c, h * w)
        """
        batch_size, channels, map_size = x.shape
        height = int(math.sqrt(map_size))
        query = torch.unsqueeze(self.mem_blocks_divider(self.w_query(x).view((batch_size, channels, height, height))), dim=4)
        key = self.mem_blocks_divider(self.w_key(x).view((batch_size, channels, height, height)))
        value = torch.unsqueeze(self.mem_blocks_divider(self.w_value(x).view((batch_size, channels, height, height))), dim=4)

        att_score = self.soft(torch.einsum('b c n g q, b c n p -> b c n g p', query, key))

        attention = torch.matmul(att_score, value)

        # no avg pooling for now
        #x = self.avg_pool(attention.view(batch_size, channels, height, height))

        #return x.view(batch_size, channels, x.shape[2] * x.shape[3])
        return attention.view(batch_size, channels, attention.shape[2] * attention.shape[3])



# init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
# init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
# init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')