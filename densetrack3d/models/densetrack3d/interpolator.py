import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int64
from torch import Tensor, nn
from torch.nn.attention import SDPBackend
from torch.nn.functional import scaled_dot_product_attention

from densetrack3d.models.densetrack3d.upsample_transformer import UpsampleCrossAttnBlock

class LearnableInterpolator(nn.Module):
    def __init__(
            self, 
            kernel_size: int = 3, 
            # stride: int = 4, 
            latent_dim: int = 128, 
            n_heads: int = 4, 
            num_attn_blocks: int = 2, 
            # upsample_factor: int = 4,
        ):
        super().__init__()

        self.kernel_size = kernel_size
        # self.stride = stride
        self.latent_dim = latent_dim
        # self.upsample_factor = upsample_factor

        self.n_heads = n_heads

        self.cross_blocks = nn.ModuleList(
            [
                UpsampleCrossAttnBlock(
                    latent_dim, 
                    latent_dim, 
                    num_heads=n_heads, 
                    mlp_ratio=4, 
                    flash=False
                )
                for _ in range(num_attn_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.final_context_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

        self.query_proj = nn.Linear(latent_dim, latent_dim, bias=True)
        self.context_proj = nn.Linear(latent_dim, latent_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        pass

    def forward(
            self, 
            feat_map: Float[Tensor, "b c1 h1 w1"],
            feat_map_up: Float[Tensor, "b c1 h2 w2"],
            upsample_factor: int = 2,
            # flow_map: Float[Tensor, "b c2 h w"],
        ):
        B = feat_map.shape[0]
        H_down, W_down = feat_map.shape[-2:]
        H_up, W_up = feat_map_up.shape[-2:]

        assert H_up == H_down * upsample_factor and W_up == W_down * upsample_factor, f"feat_map_up shape {feat_map_up.shape} is not compatible with feat_map shape {feat_map.shape} and upsample_factor {upsample_factor}"

        feat_map_down = feat_map


        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.pad(context, (0, 1, 0, 1), "replicate")

        context = F.unfold(context, kernel_size=self.kernel_size) # B C*kernel**2 H W
        context = rearrange(context, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4


        context = rearrange(context, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        x = rearrange(x, 'b c h w -> (b h w) 1 c')

        for lvl in range(len(self.cross_blocks)):
            x = self.cross_blocks[lvl](x, context)

        x = self.query_proj(self.final_norm(x))
        context = self.context_proj(self.final_context_norm(context))

        scale =  1.0 / math.sqrt(x.shape[-1])
        attn_mask = x @ context.permute(0,2,1) * scale # (b h w) 1 (i j)
        attn_mask = F.softmax(attn_mask, dim=-1)

        mask_out = rearrange(attn_mask, '(b h w) 1 c -> b c h w', h=H_up, w=W_up)

        return mask_out

    def forward_custom(
            self, 
            feat_map_context: Float[Tensor, "b c1 h1 w1 (i j)"],
            feat_map_up: Float[Tensor, "b c1 h1 w1"],
            upsample_factor: int = 2,
        ):
        B = feat_map_up.shape[0]
        H_up, W_up = feat_map_up.shape[-2:]

        context = rearrange(feat_map_context, 'b (h w) (i j) c -> (b h w) (i j) c', h=H_up, w=W_up, i=self.kernel_size, j=self.kernel_size)
        
        # NOTE prepare queries (high-reso feat)
        x = rearrange(feat_map_up, 'b c h w -> (b h w) 1 c')

        for lvl in range(len(self.cross_blocks)):
            x = self.cross_blocks[lvl](x, context)

        x = self.query_proj(self.final_norm(x))
        context = self.context_proj(self.final_context_norm(context))

        scale =  1.0 / math.sqrt(x.shape[-1])
        attn_mask = x @ context.permute(0,2,1) * scale # (b h w) 1 (i j)
        attn_mask = F.softmax(attn_mask, dim=-1)

        mask_out = rearrange(attn_mask, '(b h w) 1 c -> b c h w', h=H_up, w=W_up)

        return mask_out