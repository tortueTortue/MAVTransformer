import torch
from torch import nn
from einops.layers.torch import Rearrange

import math

cifar_10_batch = torch.rand((1,3,32,32))

small_pic_batch = torch.ones((1,1,6,6))
small_pic_batch = small_pic_batch.view((1,1,6*6))

w_query = nn.Linear(6*6,6*6, bias=False)
w_key = nn.Linear(6*6,6*6, bias=False)
w_value = nn.Linear(6*6,6*6, bias=False)

w_query.eval()
w_key.eval()
w_value.eval()

# att 
soft = nn.Softmax(dim=4)

mem_blocks_divider = Rearrange('b c (h p1) (w p2) -> b c (h w) (p1 p2)', p1=2,p2=2)

# TODO Optimize
query =  mem_blocks_divider(w_query(small_pic_batch).view((1,1,6,6))).view((1,1,9,4,1))
key   =  mem_blocks_divider(w_key(small_pic_batch).view((1,1,6,6)))
value =  mem_blocks_divider(w_value(small_pic_batch).view((1,1,6,6))).view((1,1,9,4,1))

# q(i,j) * k(a,b) where i,j are pixel col and row AND a,b are memory block size
att_score = soft(torch.einsum('b c n g q, b c n p ->b c n g p', query, key))

print(torch.matmul(att_score, value).view(1,1,9*4))

# avg pooling

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

    def forward(self, x):
        """
        Here we assume that x is in the shape (b, c, h * w)
        """

        query = torch.unsqueeze(self.mem_blocks_divider(self.w_query(x)), dim=4)
        key = self.mem_blocks_divider(self.w_key(x))
        value = torch.unsqueeze(self.mem_blocks_divider(self.w_value(x)), dim=4)

        att_score = soft(torch.einsum('b c n g q, b c n p -> b c n g p', query, key))

        attention = torch.matmul(att_score, value)

        # no avg pooling for now

        return attention.views(-1, attention.shape[1], attention.shape[2] * attention.shape[3])