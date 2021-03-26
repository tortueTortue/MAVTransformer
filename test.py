from torch import nn
import torch

batch_1 = torch.rand((30,3,50,50))

print(f"In shape {batch_1.shape}")

# The output of this conv layer should be (30, 1, 10, 10)
print(f"Out shape first conv {nn.Conv2d(3, 1, 5, stride = 5, bias = False).forward(batch_1).shape}")

# The output of this conv layer should be (30, 1, 50, 50)
print(f"Out shape second conv {nn.Conv2d(3, 1, 1, stride = 1, bias = False).forward(batch_1).shape}")

# The output of this conv layer should be (30, 1, 50, 50)
print(f"Out shape third conv {nn.Conv2d(3, 1, 1, stride = 1, bias = False).forward(batch_1).shape}")

pic = torch.ones((1,1,5,5))
pic = pic.view((1,1,5*5))

lin = nn.Linear(5*5,5*5, bias=False)

lin.weight.data.fill_(3)

print(f"Pic : {pic}")
print(f"Pic shape : {pic.shape}")

out = lin(pic)
print(f"Out pic : {out}")
print(f"Out pic shape : {out.shape}")

