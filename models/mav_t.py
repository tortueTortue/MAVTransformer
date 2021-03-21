"""
MAViT : Mixed Attention Vision Transformer
T2T - MAViT : Mixed Attention Vision Transformer

MAViT is a mixed of two types of attention modules for vision : 
    - Attention applied to patches of pictures
    - Local Attention applied memory blocks (pixels zone)

To make it simple, we'll separate the memory blocks the same the in the patches are divided
We'll try two variants with this model : Patches --> Local blocks and Local blocks --> Patches

Then we'll try the best of the two variants with T2T dim reduction blocks
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.ViT import ViT
from models.LAT import LAT

class MAViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, no_of_blocks, heads,
                 mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., is_vit_first: bool = True):
        super(MAViT, self).__init__()
        self.is_vit_first = is_vit_first
        self.ViT = ViT(image_size, patch_size, num_classes, dim, no_of_blocks, heads,
                        mlp_dim, pool, channels, dim_head, dropout,
                        emb_dropout)

        self.LAT = LAT(dim, no_of_blocks, mlp_dim, dropout = dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask = None):
        x = self.LAT(self.ViT(x)) if self.is_vit_first else self.ViT(self.LAT(x))

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        
        return self.mlp_head(x)