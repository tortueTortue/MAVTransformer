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

import math

class MAViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, no_of_blocks, heads,
                 mlp_dim, memory_block_size=4, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., is_vit_first: bool = True, batch_size = 128, kernel_size=2, max_pooling_blocks=3):
        """
        args:
            dim : output vector length (in the case of ViT, it is the length of 
                                        the patches after going through the first linear)
        """
        super(MAViT, self).__init__()
        self.is_vit_first = is_vit_first
        self.ViT = ViT(image_size, patch_size, num_classes, dim, no_of_blocks, heads,
                       mlp_dim, pool, channels, dim_head, dropout, emb_dropout)

        # Reshape patches into image
        self.no_of_patches = int((image_size / patch_size) ** 2)
        
        mlp_head_size = 0

        if is_vit_first:
            self.LAT = LAT(image_size**2, no_of_blocks, dropout = dropout,
                            memory_block_size=memory_block_size, kernel_size=kernel_size)
            mlp_head_size = (image_size**2) // kernel_size ** (2 * (no_of_blocks-max_pooling_blocks))
        else :
            self.LAT = LAT(image_size**2, no_of_blocks, dropout = dropout, max_pooling_blocks=0,
                           memory_block_size=memory_block_size, kernel_size=kernel_size)
            mlp_head_size = channels * (patch_size**2)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.class_token_pool = nn.AvgPool1d(kernel_size=(dim//mlp_head_size))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_head_size),
            nn.Linear(mlp_head_size, num_classes)
        )

    def forward(self, x, mask = None):
        batch_size, channels, h, w = x.shape

        if self.is_vit_first:
            x = self.ViT(x)
            class_token, x = torch.split(x, [1, self.no_of_patches], dim=1)
            x = self.LAT(x.view(batch_size, channels, h * w))
            class_token = self.class_token_pool(class_token)
            x = torch.cat((class_token, x), dim=1)
        else:
            x = self.LAT(x.view(batch_size, channels, h * w))
            height = int(math.sqrt(x.shape[2]))
            x = self.ViT(x.view(batch_size, channels, height, height))

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        
        return self.mlp_head(x)