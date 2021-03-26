from torch.nn import Module

class PatchToImage(Module):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)