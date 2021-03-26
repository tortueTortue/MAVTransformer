import torch
from torch import nn
from einops.layers.torch import Rearrange

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

mem_blocks_divider = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2,p2=2)

# TODO Optimize
query =  mem_blocks_divider(w_query(small_pic_batch.view((1,1,6*6)).view((1,1,6,6))).view((1,1,9,4,1))
key   =  mem_blocks_divider(w_key  (small_pic_batch.view((1,1,6*6)).view((1,1,6,6)))
value =  mem_blocks_divider(w_value(small_pic_batch.view((1,1,6*6)).view((1,1,6,6))).view((1,1,9,4,1))

# q(i,j) * k(a,b) where i,j are pixel col and row AND a,b are memory block size
att_score = soft(torch.einsum('b c n g q, b c n p -> n g p', q,k))

torch.matmul(att_score, value).view(1,1,9*4)