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


from densetrack3d.models.densetrack3d.blocks import (
    Attention,
    AttnBlock,
    BasicEncoder,
    CorrBlock,
    Mlp,
    ResidualBlock,
    ResidualBlock1d,
    Conv1dPad,
    Upsample,
    cam2pix,
    pix2cam,
)
from densetrack3d.models.embeddings import (
    get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_embedding,
    get_2d_sincos_pos_embed,
    get_3d_embedding,
    get_3d_embedding_custom,
)
from densetrack3d.models.model_utils import (
    bilinear_sampler,
    depth_to_disparity,
    get_grid,
    get_points_on_a_grid,
    sample_features4d,
    sample_features5d,
)


class RelativeAttention(nn.Module):
    """Multi-headed attention (MHA) module."""

    def __init__(self, query_dim, num_heads=8, qkv_bias=True, model_size=None, flash=False):
        super(RelativeAttention, self).__init__()

        query_dim = query_dim // num_heads
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.value_size = query_dim
        self.model_size = query_dim * num_heads

        self.qkv_bias = qkv_bias

        self.flash = flash

        self.query_proj = nn.Linear(num_heads * query_dim, num_heads * query_dim, bias=qkv_bias)
        self.key_proj = nn.Linear(num_heads * query_dim, num_heads * query_dim, bias=qkv_bias)
        self.value_proj = nn.Linear(num_heads * self.value_size, num_heads * self.value_size, bias=qkv_bias)
        self.final_proj = nn.Linear(num_heads * self.value_size, self.model_size, bias=qkv_bias)

        self.scale = 1.0 / math.sqrt(self.query_dim)
        # self.training_length = 24

        # bias_forward = get_alibi_slope(self.num_heads // 2) * get_relative_positions(self.training_length)
        # bias_forward = bias_forward + torch.triu(torch.full_like(bias_forward, -1e9), diagonal=1)
        # bias_backward = get_alibi_slope(self.num_heads // 2) * get_relative_positions(self.training_length, reverse=True)
        # bias_backward = bias_backward + torch.tril(torch.full_like(bias_backward, -1e9), diagonal=-1)

        # self.register_buffer("precomputed_attn_bias", torch.cat([bias_forward, bias_backward], dim=0), persistent=False)

    def forward(self, x, context, attn_bias=None):
        B, N1, C = x.size()

        q = self._linear_projection(x, self.query_dim, self.query_proj)  # [T', H, Q=K]
        k = self._linear_projection(context, self.query_dim, self.key_proj)  # [T, H, K]
        v = self._linear_projection(context, self.value_size, self.value_proj)  # [T, H, V]

        if self.flash:
            # with torch.autocast(device_type="cuda", enabled=True):
            #     x = flash_attn_func(q.half(), k.half(), v.half())
            #     x = x.reshape(B, N1, C)
            # x = x.float()
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                dtype = k.dtype
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    q = q.to(torch.bfloat16)
                    k = k.to(torch.bfloat16)
                    v = v.to(torch.bfloat16)
                    x = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False, scale=self.scale)
                if dtype == torch.float32:  # if input was FP32, cast back to FP32
                    x = x.to(torch.float32)
                x = x.transpose(1, 2).reshape(B, N1, C)
        else:
            q = q.permute(0, 2, 1, 3) # B H N C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            sim = (q @ k.transpose(-2, -1)) * self.scale

            if attn_bias is not None:
                sim = sim + attn_bias
            attn = sim.softmax(dim=-1)

            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N1, C)

        # with torch.autocast(device_type="cuda", dtype=torch.float32):
        #     attn = F.scaled_dot_product_attention(query_heads, key_heads, value_heads, attn_mask=attn_bias, scale=1.0 / math.sqrt(self.query_dim))
        # else:

        #     sim = (query_heads @ key_heads.transpose(-2, -1)) * self.scale

        #     if attn_bias is not None:
        #         sim = sim + attn_bias
        #     attn = sim.softmax(dim=-1)

        #     attn = (attn @ value_heads)
        # attn = attn.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)

        return self.final_proj(x)  # [T', D']

    def _linear_projection(self, x, head_size, proj_layer):
        batch_size, sequence_length, _ = x.shape
        y = proj_layer(x)
        y = y.reshape((batch_size, sequence_length, self.num_heads, head_size))

        return y


class UpsampleCrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = RelativeAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, attn_bias=None):
        x = x + self.cross_attn(x=self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderUpsampler(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int = None, stride: int = 4):
        super().__init__()

        self.stride = stride

        if out_channels is None:
            out_channels = middle_channels

        self.conv_in = nn.Conv2d(in_channels, middle_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.norm1 = nn.GroupNorm(num_groups=middle_channels // 8, num_channels=middle_channels, eps=1e-6)

        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for i in range(int(math.log2(self.stride))):
            self.res_blocks.append(ResidualBlock(middle_channels, middle_channels))
            self.upsample_blocks.append(Upsample(middle_channels, with_conv=True))

            # in_channels = middle_channels

        self.norm2 = nn.GroupNorm(num_groups=middle_channels // 8, num_channels=middle_channels, eps=1e-6)
        self.conv_out = nn.Conv2d(middle_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.initialize_weight()

    def initialize_weight(self):
        def _basic_init(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.res_blocks.apply(_basic_init)
        self.conv_in.apply(_basic_init)
        self.conv_out.apply(_basic_init)

    def forward(
        self,
        x: Float[Tensor, "b c1 h_down w_down"],
        mode: str = "nearest",
    ) -> Float[Tensor, "b c1 h_up w_up"]:

        x = F.relu(self.norm1(self.conv_in(x)))

        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
            x = self.upsample_blocks[i](x, mode=mode)

        x = self.conv_out(F.relu(self.norm2(x)))
        return x


class UpsampleTransformer(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 4,
        latent_dim: int = 128,
        n_heads: int = 4,
        num_attn_blocks: int = 2,
        use_rel_emb: bool = True,
        flash: bool = False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim

        self.n_heads = n_heads

        self.attnup_feat_cnn = DecoderUpsampler(
            in_channels=self.latent_dim, middle_channels=self.latent_dim, out_channels=self.latent_dim
        )

        self.cross_blocks = nn.ModuleList(
            [
                UpsampleCrossAttnBlock(latent_dim + 64, latent_dim + 64, num_heads=n_heads, mlp_ratio=4, flash=flash)
                for _ in range(num_attn_blocks)
            ]
        )

        self.flow_mlp = nn.Sequential(
            nn.Conv2d(2 * 16, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Linear(latent_dim + 64, kernel_size * kernel_size, bias=True)

        if use_rel_emb:
            self.rpb_attnup = nn.Parameter(torch.zeros(kernel_size * kernel_size))
            torch.nn.init.trunc_normal_(self.rpb_attnup, std=0.1, mean=0.0, a=-2.0, b=2.0)
        else:
            self.rpb_attnup = None

    def forward(
        self,
        feat_map: Float[Tensor, "b c1 h w"],
        flow_map: Float[Tensor, "b c2 h w"],
    ):
        B = feat_map.shape[0]
        H_down, W_down = feat_map.shape[-2:]
        # x0, y0 = x0y0

        feat_map_up = self.attnup_feat_cnn(feat_map)  # learnable upsample by 4
        # feat_map_down = F.interpolate(feat_map_up, scale_factor=1/self.stride, mode='nearest') # B C H*4 W*4
        feat_map_down = feat_map
        # depths_down = F.interpolate(depths, scale_factor=1/self.stride, mode='nearest')

        # NOTE prepare attention bias
        # depths_down_ = torch.stack([depths_down[b, :, y0_:y0_+H_down, x0_:x0_+W_down] for b, (x0_,y0_) in enumerate(zip(x0, y0))], dim=0)
        # depths_ = torch.stack([depths[b, :, y0_*4:y0_*4+H_down*4, x0_*4:x0_*4+W_down*4] for b, (x0_,y0_) in enumerate(zip(x0, y0))], dim=0)
        # guidance_downsample = F.interpolate(guidance, size=(H, W), mode='nearest')
        pad_val = (self.kernel_size - 1) // 2
        # depths_down_padded = F.pad(depths_down_, (pad_val, pad_val, pad_val, pad_val), "replicate")
        # unfold_depths_down_padded = F.unfold(depths_down_padded, [self.kernel_size, self.kernel_size], padding=0).reshape(B, self.kernel_size**2, H_down, W_down)
        # unfold_depths_down_padded = F.interpolate(unfold_depths_down_padded, scale_factor=self.stride, mode='nearest')

        # disps_ = depth_to_disparity(depths_, d_near, d_far) * Dz
        # unfold_disps_down_padded = depth_to_disparity(unfold_depths_down_padded, d_near, d_far) * Dz
        # relative_disps = torch.abs(disps_ - unfold_disps_down_padded) # B 9 H W

        if self.rpb_attnup is not None:
            relative_pos_attn_map = self.rpb_attnup.view(1, 1, -1, 1, 1).repeat(
                B, self.n_heads, 1, H_down * 4, W_down * 4
            )
            relative_pos_attn_map = rearrange(relative_pos_attn_map, "b k n h w -> (b h w) k 1 n")
            attn_bias = relative_pos_attn_map
        else:
            attn_bias = None

        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.unfold(context, kernel_size=self.kernel_size, padding=pad_val)  # B C*kernel**2 H W
        context = rearrange(context, "b c (h w) -> b c h w", h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=self.stride, mode="nearest")  # B C*kernel**2 H*4 W*4
        context = rearrange(context, "b (c i j) h w -> (b h w) (i j) c", i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        x = rearrange(x, "b c h w -> (b h w) 1 c")

        assert flow_map.shape[-2:] == feat_map.shape[-2:]

        flow_map = rearrange(flow_map, "b t c h w -> b (t c) h w")
        flow_map = self.flow_mlp(flow_map)

        nn_flow_map = F.unfold(flow_map, kernel_size=self.kernel_size, padding=pad_val)  # B C*kernel**2 H W
        nn_flow_map = rearrange(nn_flow_map, "b c (h w) -> b c h w", h=H_down, w=W_down)
        nn_flow_map = F.interpolate(nn_flow_map, scale_factor=self.stride, mode="nearest")  # B C*kernel**2 H*4 W*4
        nn_flow_map = rearrange(
            nn_flow_map, "b (c i j) h w -> (b h w) (i j) c", i=self.kernel_size, j=self.kernel_size
        )

        up_flow_map = F.interpolate(flow_map, scale_factor=4, mode="nearest")  # NN up # b 2 h w
        up_flow_map = rearrange(up_flow_map, "b c h w -> (b h w) 1 c")

        context = torch.cat([context, nn_flow_map], dim=-1)
        x = torch.cat([x, up_flow_map], dim=-1)

        for lvl in range(len(self.cross_blocks)):
            x = self.cross_blocks[lvl](x, context, attn_bias)

        mask_out = self.out(x)
        mask_out = F.softmax(mask_out, dim=-1)
        mask_out = rearrange(mask_out, "(b h w) 1 c -> b c h w", h=H_down * self.stride, w=W_down * self.stride)

        return mask_out


def get_alibi_slope(num_heads):
    x = (24) ** (1 / num_heads)
    return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)]).float()


class UpsampleTransformerAlibi(nn.Module):
    def __init__(
            self, 
            kernel_size: int = 3, 
            stride: int = 4, 
            latent_dim: int = 128, 
            n_heads: int = 4, 
            num_attn_blocks: int = 2, 
            upsample_factor: int = 4,
        ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim
        self.upsample_factor = upsample_factor

        self.n_heads = n_heads

        self.attnup_feat_cnn = DecoderUpsampler(
            in_channels=self.latent_dim,
            middle_channels=self.latent_dim,
            out_channels=self.latent_dim,
            # stride=self.upsample_factor
        )

        self.cross_blocks = nn.ModuleList(
            [
                UpsampleCrossAttnBlock(
                    latent_dim+64, 
                    latent_dim+64, 
                    num_heads=n_heads, 
                    mlp_ratio=4, 
                    flash=False
                )
                for _ in range(num_attn_blocks)
            ]
        )

        self.flow_mlp = nn.Sequential(
            nn.Conv2d(2*16, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Linear(latent_dim+64, kernel_size*kernel_size, bias=True)
        

        alibi_slope = get_alibi_slope(n_heads // 2)
        grid_kernel = get_grid(kernel_size, kernel_size, normalize=False).reshape(kernel_size, kernel_size, 2)
        grid_kernel = grid_kernel - (kernel_size - 1) / 2
        grid_kernel = -torch.abs(grid_kernel)
        alibi_bias = torch.cat([
            alibi_slope.view(-1,1,1) * grid_kernel[..., 0].view(1,kernel_size,kernel_size),
            alibi_slope.view(-1,1,1) * grid_kernel[..., 1].view(1,kernel_size,kernel_size)
        ]) # n_heads, kernel_size, kernel_size

        self.register_buffer("alibi_bias", alibi_bias)

        self._init_weights()

    def _init_weights(self):
        self.out.weight.data.fill_(0)
        self.out.bias.data.fill_(0)

    def forward(
            self, 
            feat_map: Float[Tensor, "b c1 h w"],
            flow_map: Float[Tensor, "b c2 h w"],
        ):
        B = feat_map.shape[0]
        H_down, W_down = feat_map.shape[-2:]

        feat_map_up = self.attnup_feat_cnn(feat_map) # learnable upsample by 4
        if self.upsample_factor != 4:
            additional_scale = float(self.upsample_factor / 4)
            if additional_scale > 1:
                feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='bilinear', align_corners=False)
            else:
                feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='nearest')

        feat_map_down = feat_map

        pad_val = (self.kernel_size - 1) // 2

        attn_bias = self.alibi_bias.view(1,self.n_heads,self.kernel_size**2,1,1).repeat(B,1,1,H_down*self.upsample_factor,W_down*self.upsample_factor)
        attn_bias = rearrange(attn_bias, "b k n h w -> (b h w) k 1 n")

        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.unfold(context, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        context = rearrange(context, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        context = rearrange(context, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        x = rearrange(x, 'b c h w -> (b h w) 1 c')


        assert flow_map.shape[-2:] == feat_map.shape[-2:]


        flow_map = rearrange(flow_map, 'b t c h w -> b (t c) h w')
        flow_map = self.flow_mlp(flow_map)

        nn_flow_map = F.unfold(flow_map, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        nn_flow_map = rearrange(nn_flow_map, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        nn_flow_map = F.interpolate(nn_flow_map, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        nn_flow_map = rearrange(nn_flow_map, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        up_flow_map = F.interpolate(flow_map, scale_factor=self.upsample_factor, mode="nearest") # NN up # b 2 h w
        up_flow_map = rearrange(up_flow_map, 'b c h w -> (b h w) 1 c')

        context = torch.cat([context, nn_flow_map], dim=-1)
        x = torch.cat([x, up_flow_map], dim=-1)

        for lvl in range(len(self.cross_blocks)):
            x = self.cross_blocks[lvl](x, context, attn_bias)

        mask_out = self.out(x)
        mask_out = F.softmax(mask_out, dim=-1)
        mask_out = rearrange(mask_out, '(b h w) 1 c -> b c h w', h=H_down*self.upsample_factor, w=W_down*self.upsample_factor)

        return mask_out

class UpsampleTransformerAlibiTemporal(nn.Module):
    def __init__(
            self, 
            kernel_size: int = 3, 
            stride: int = 4, 
            latent_dim: int = 128, 
            n_heads: int = 4, 
            num_attn_blocks: int = 2, 
            upsample_factor: int = 4,
        ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim
        self.upsample_factor = upsample_factor

        self.n_heads = n_heads

        self.attnup_feat_cnn = DecoderUpsampler(
            in_channels=self.latent_dim,
            middle_channels=self.latent_dim,
            out_channels=self.latent_dim,
            # stride=self.upsample_factor
        )

        self.cross_blocks = nn.ModuleList(
            [
                UpsampleCrossAttnBlock(
                    latent_dim+64, 
                    latent_dim+64, 
                    num_heads=n_heads, 
                    mlp_ratio=4, 
                    flash=False
                )
                for _ in range(num_attn_blocks)
            ]
        )

        # self.time_blocks = nn.ModuleList(
        #     [
        #         AttnBlock(
        #             latent_dim+64,
        #             n_heads,
        #             mlp_ratio=4,
        #             attn_class=Attention,
        #             flash=False,
        #         )
        #         for _ in range(num_attn_blocks)
        #     ]
        # )

        self.flow_mlp = nn.Sequential(
            nn.Conv2d(2, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Linear(latent_dim+64, kernel_size*kernel_size, bias=True)
        

        alibi_slope = get_alibi_slope(n_heads // 2)
        grid_kernel = get_grid(kernel_size, kernel_size, normalize=False).reshape(kernel_size, kernel_size, 2)
        grid_kernel = grid_kernel - (kernel_size - 1) / 2
        grid_kernel = -torch.abs(grid_kernel)
        alibi_bias = torch.cat([
            alibi_slope.view(-1,1,1) * grid_kernel[..., 0].view(1,kernel_size,kernel_size),
            alibi_slope.view(-1,1,1) * grid_kernel[..., 1].view(1,kernel_size,kernel_size)
        ]) # n_heads, kernel_size, kernel_size

        self.register_buffer("alibi_bias", alibi_bias)

        self._init_weights()

    def _init_weights(self):
        self.out.weight.data.fill_(0)
        self.out.bias.data.fill_(0)

    def forward(
            self, 
            feat_map: Float[Tensor, "b t c1 h w"],
            flow_map: Float[Tensor, "b t c2 h w"],
        ):

        # NOTE hardcode subsample here
        temporal_sample_inds = [0, 5, 10, 15]

        # breakpoint()
        feat_map = feat_map[:, temporal_sample_inds, :, :, :]
        flow_map = flow_map[:, temporal_sample_inds, :, :, :]
        # breakpoint()

        B, T = feat_map.shape[:2]
        H_down, W_down = feat_map.shape[-2:]

        feat_map = rearrange(feat_map, 'b t c h w -> (b t) c h w')
        flow_map = rearrange(flow_map, 'b t c h w -> (b t) c h w')

        feat_map_up = self.attnup_feat_cnn(feat_map) # learnable upsample by 4
        if self.upsample_factor != 4:
            additional_scale = float(self.upsample_factor / 4)
            if additional_scale > 1:
                feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='bilinear', align_corners=False)
            else:
                feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='nearest')


        H_up, W_up = feat_map_up.shape[-2:]

        feat_map_down = feat_map

        pad_val = (self.kernel_size - 1) // 2

        attn_bias = self.alibi_bias.view(1,self.n_heads,self.kernel_size**2,1,1).repeat(B*T,1,1,H_down*self.upsample_factor,W_down*self.upsample_factor)
        attn_bias = rearrange(attn_bias, "b k n h w -> (b h w) k 1 n")

        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.unfold(context, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        context = rearrange(context, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        # context = rearrange(context, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)
        context = rearrange(context, '(b t) (c i j) h w -> (b h w t) (i j) c', b=B, t=T, i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        # x = rearrange(x, 'b c h w -> (b h w) 1 c')
        x = rearrange(x, '(b t) c h w -> (b h w t) 1 c', b=B, t=T)


        assert flow_map.shape[-2:] == feat_map.shape[-2:]


        # flow_map = rearrange(flow_map, 'b t c h w -> b (t c) h w')
        flow_map = self.flow_mlp(flow_map)

        nn_flow_map = F.unfold(flow_map, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        nn_flow_map = rearrange(nn_flow_map, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        nn_flow_map = F.interpolate(nn_flow_map, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        # nn_flow_map = rearrange(nn_flow_map, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)
        nn_flow_map = rearrange(nn_flow_map, '(b t) (c i j) h w -> (b h w t) (i j) c', b=B, t=T, i=self.kernel_size, j=self.kernel_size)

        up_flow_map = F.interpolate(flow_map, scale_factor=self.upsample_factor, mode="nearest") # NN up # b 2 h w
        # up_flow_map = rearrange(up_flow_map, 'b c h w -> (b h w) 1 c')
        up_flow_map = rearrange(up_flow_map, '(b t) c h w -> (b h w t) 1 c', b=B, t=T)

        context = torch.cat([context, nn_flow_map], dim=-1)
        x = torch.cat([x, up_flow_map], dim=-1)

        # breakpoint()

        for lvl in range(len(self.cross_blocks)):
            # NOTE cross attn
            x = self.cross_blocks[lvl](x, context, attn_bias)


            # breakpoint()
            # # NOTE temporal attn
            # x = rearrange(x, '(b h w t) 1 c -> (b h w) t c', b=B, t=T, h=H_up, w=W_up)
            # x = self.time_blocks[lvl](x)

            # # breakpoint()
            # x = rearrange(x, '(b h w) t c -> (b h w t) 1 c', b=B, h=H_up, w=W_up)

        x = rearrange(x, '(b h w t) 1 c -> (b h w) t c', b=B, t=T, h=H_up, w=W_up)
        # breakpoint()
        x = x.mean(dim=1) # average over time

        # breakpoint()
        mask_out = self.out(x)
        mask_out = F.softmax(mask_out, dim=-1)
        mask_out = rearrange(mask_out, '(b h w) c -> b c h w', b=B, h=H_down*self.upsample_factor, w=W_down*self.upsample_factor)

        return mask_out


class UpsampleTransformerAlibiV2(nn.Module):
    def __init__(
            self, 
            kernel_size: int = 3, 
            stride: int = 4, 
            latent_dim: int = 128, 
            n_heads: int = 4, 
            num_attn_blocks: int = 2, 
            upsample_factor: int = 4,
        ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim
        self.upsample_factor = upsample_factor

        self.n_heads = n_heads

        self.attnup_feat_cnn = DecoderUpsampler(
            in_channels=self.latent_dim,
            middle_channels=self.latent_dim,
            out_channels=self.latent_dim,
            # stride=self.upsample_factor
        )

        self.cross_blocks = nn.ModuleList(
            [
                UpsampleCrossAttnBlock(
                    latent_dim+64, 
                    latent_dim+64, 
                    num_heads=n_heads, 
                    mlp_ratio=4, 
                    flash=False
                )
                for _ in range(num_attn_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(latent_dim+64, elementwise_affine=False, eps=1e-6)
        self.final_context_norm = nn.LayerNorm(latent_dim+64, elementwise_affine=False, eps=1e-6)

        self.flow_mlp = nn.Sequential(
            nn.Conv2d(2*16, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.out = Mlp(
            in_features=3*(latent_dim+64),
            hidden_features=3*(latent_dim+64),
            out_features=1
        )
        
        # nn.Linear(
        #     latent_dim+64, 
        #     kernel_size*kernel_size, 
        #     bias=True
        # )
        

        alibi_slope = get_alibi_slope(n_heads // 2)
        grid_kernel = get_grid(kernel_size, kernel_size, normalize=False).reshape(kernel_size, kernel_size, 2)
        grid_kernel = grid_kernel - (kernel_size - 1) / 2
        grid_kernel = -torch.abs(grid_kernel)
        alibi_bias = torch.cat([
            alibi_slope.view(-1,1,1) * grid_kernel[..., 0].view(1,kernel_size,kernel_size),
            alibi_slope.view(-1,1,1) * grid_kernel[..., 1].view(1,kernel_size,kernel_size)
        ]) # n_heads, kernel_size, kernel_size

        self.register_buffer("alibi_bias", alibi_bias)

        self._init_weights()

    def _init_weights(self):
        pass
        # self.out.fc2.weight.data.fill_(0)
        # self.out.fc2.bias.data.fill_(0)

    def forward(
            self, 
            feat_map: Float[Tensor, "b c1 h w"],
            flow_map: Float[Tensor, "b c2 h w"],
        ):
        B = feat_map.shape[0]
        H_down, W_down = feat_map.shape[-2:]

        feat_map_up = self.attnup_feat_cnn(feat_map) # learnable upsample by 4
        if self.upsample_factor != 4:
            additional_scale = float(self.upsample_factor / 4)
            if additional_scale > 1:
                feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='bilinear', align_corners=False)
            else:
                feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='nearest')

        feat_map_down = feat_map

        pad_val = (self.kernel_size - 1) // 2

        attn_bias = self.alibi_bias.view(1,self.n_heads,self.kernel_size**2,1,1).repeat(B,1,1,H_down*self.upsample_factor,W_down*self.upsample_factor)
        attn_bias = rearrange(attn_bias, "b k n h w -> (b h w) k 1 n")

        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.unfold(context, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        context = rearrange(context, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        context = rearrange(context, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        x = rearrange(x, 'b c h w -> (b h w) 1 c')


        assert flow_map.shape[-2:] == feat_map.shape[-2:]


        flow_map = rearrange(flow_map, 'b t c h w -> b (t c) h w')
        flow_map = self.flow_mlp(flow_map)

        nn_flow_map = F.unfold(flow_map, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        nn_flow_map = rearrange(nn_flow_map, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        nn_flow_map = F.interpolate(nn_flow_map, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        nn_flow_map = rearrange(nn_flow_map, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        up_flow_map = F.interpolate(flow_map, scale_factor=self.upsample_factor, mode="nearest") # NN up # b 2 h w
        up_flow_map = rearrange(up_flow_map, 'b c h w -> (b h w) 1 c')

        context = torch.cat([context, nn_flow_map], dim=-1)
        x = torch.cat([x, up_flow_map], dim=-1)

        for lvl in range(len(self.cross_blocks)):
            x = self.cross_blocks[lvl](x, context, attn_bias)

        x = self.final_norm(x)
        context = self.final_context_norm(context)
        final_input = torch.cat([
            x.repeat(1,self.kernel_size**2,1),
            context,
            x - context
        ], dim=-1) # (b h w) (i j) c

        final_output = self.out(final_input) 
        final_output = rearrange(final_output, '(b h w) c 1 -> b c h w', h=H_down*self.upsample_factor, w=W_down*self.upsample_factor)
        mask_out = F.softmax(final_output, dim=1)

        # mask_out = self.out(x)
        # mask_out = F.softmax(mask_out, dim=-1)
        # mask_out = rearrange(mask_out, '(b h w) 1 c -> b c h w', h=H_down*self.upsample_factor, w=W_down*self.upsample_factor)

        return mask_out


class UpsampleTransformerAlibiV3(nn.Module):
    def __init__(
            self, 
            kernel_size: int = 3, 
            stride: int = 4, 
            latent_dim: int = 128, 
            n_heads: int = 4, 
            num_attn_blocks: int = 2, 
            upsample_factor: int = 4,
        ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim
        self.upsample_factor = upsample_factor

        self.n_heads = n_heads

        self.attnup_feat_cnn = DecoderUpsampler(
            in_channels=self.latent_dim,
            middle_channels=self.latent_dim,
            out_channels=self.latent_dim,
            # stride=self.upsample_factor
        )

        self.cross_blocks = nn.ModuleList(
            [
                UpsampleCrossAttnBlock(
                    latent_dim+64, 
                    latent_dim+64, 
                    num_heads=n_heads, 
                    mlp_ratio=4, 
                    flash=False
                )
                for _ in range(num_attn_blocks)
            ]
        )

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    latent_dim+64,
                    n_heads,
                    mlp_ratio=4,
                    attn_class=Attention,
                    flash=False,
                )
                for _ in range(num_attn_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(latent_dim+64, elementwise_affine=False, eps=1e-6)
        self.final_context_norm = nn.LayerNorm(latent_dim+64, elementwise_affine=False, eps=1e-6)

        self.flow_mlp = nn.Sequential(
            nn.Conv2d(2, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.out = Mlp(
            in_features=3*(latent_dim+64),
            hidden_features=3*(latent_dim+64),
            out_features=1
        )
        
        

        alibi_slope = get_alibi_slope(n_heads // 2)
        grid_kernel = get_grid(kernel_size, kernel_size, normalize=False).reshape(kernel_size, kernel_size, 2)
        grid_kernel = grid_kernel - (kernel_size - 1) / 2
        grid_kernel = -torch.abs(grid_kernel)
        alibi_bias = torch.cat([
            alibi_slope.view(-1,1,1) * grid_kernel[..., 0].view(1,kernel_size,kernel_size),
            alibi_slope.view(-1,1,1) * grid_kernel[..., 1].view(1,kernel_size,kernel_size)
        ]) # n_heads, kernel_size, kernel_size

        self.register_buffer("alibi_bias", alibi_bias)

        self._init_weights()

    def _init_weights(self):
        pass
        # self.out.fc2.weight.data.fill_(0)
        # self.out.fc2.bias.data.fill_(0)

    def forward(
            self, 
            feat_map: Float[Tensor, "b t c1 h w"],
            flow_map: Float[Tensor, "b t c2 h w"],
        ):
        B, T = feat_map.shape[:2]
        H_down, W_down = feat_map.shape[-2:]

        feat_map = rearrange(feat_map, 'b t c h w -> (b t) c h w')
        flow_map = rearrange(flow_map, 'b t c h w -> (b t) c h w')

        feat_map_up = self.attnup_feat_cnn(
            feat_map
        ) # learnable upsample by 4
        if self.upsample_factor != 4:
            additional_scale = float(self.upsample_factor / 4)
            if additional_scale > 1:
                feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='bilinear', align_corners=False)
            else:
                feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='nearest')


        H_up, W_up = feat_map_up.shape[-2:]

        feat_map_down = feat_map

        pad_val = (self.kernel_size - 1) // 2

        attn_bias = self.alibi_bias.view(1,self.n_heads,self.kernel_size**2,1,1).repeat(B,1,1,H_down*self.upsample_factor,W_down*self.upsample_factor)
        attn_bias = rearrange(attn_bias, "b k n h w -> (b h w) k 1 n")

        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.unfold(context, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        context = rearrange(context, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        # context = rearrange(context, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)
        context = rearrange(context, '(b t) (c i j) h w -> (b h w t) (i j) c', b=B, t=T, i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        # x = rearrange(x, 'b c h w -> (b h w) 1 c')
        x = rearrange(x, '(b t) c h w -> (b h w t) 1 c', b=B, t=T)


        assert flow_map.shape[-2:] == feat_map.shape[-2:]


        # flow_map = rearrange(flow_map, 'b t c h w -> b (t c) h w')
        flow_map = self.flow_mlp(flow_map)

        nn_flow_map = F.unfold(flow_map, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        nn_flow_map = rearrange(nn_flow_map, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        nn_flow_map = F.interpolate(nn_flow_map, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        # nn_flow_map = rearrange(nn_flow_map, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)
        nn_flow_map = rearrange(nn_flow_map, '(b t) (c i j) h w -> (b h w t) (i j) c', b=B, t=T, i=self.kernel_size, j=self.kernel_size)

        up_flow_map = F.interpolate(flow_map, scale_factor=self.upsample_factor, mode="nearest") # NN up # b 2 h w
        # up_flow_map = rearrange(up_flow_map, 'b c h w -> (b h w) 1 c')
        up_flow_map = rearrange(up_flow_map, '(b t) c h w -> (b h w t) 1 c', b=B, t=T)

        context = torch.cat([context, nn_flow_map], dim=-1)
        x = torch.cat([x, up_flow_map], dim=-1)

        for lvl in range(len(self.cross_blocks)):
            # NOTE cross attn
            x = self.cross_blocks[lvl](x, context, attn_bias)
            # NOTE temporal attn
            x = rearrange(x, '(b h w t) 1 c -> (b h w) t c', b=B, t=T, h=H_up, w=W_up)
            x = self.time_blocks[lvl](x)
            x = rearrange('(b h w) t c -> (b h w t) 1 c', b=B, t=T, h=H_up, w=W_up)

        x = self.final_norm(x)
        context = self.final_context_norm(context)
        final_input = torch.cat([
            x.repeat(1,self.kernel_size**2,1),
            context,
            x - context
        ], dim=-1) # (b h w) (i j) c

        final_output = self.out(final_input) 
        final_output = rearrange(final_output, '(b h w) c 1 -> b c h w', h=H_down*self.upsample_factor, w=W_down*self.upsample_factor)
        mask_out = F.softmax(final_output, dim=1)
        return mask_out




class UpsampleTransformerV2(nn.Module):
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

        # self.attnup_feat_cnn = DecoderUpsampler(
        #     in_channels=self.latent_dim,
        #     middle_channels=self.latent_dim,
        #     out_channels=self.latent_dim,
        #     # stride=self.upsample_factor
        # )

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

        # self.flow_mlp = nn.Sequential(
        #     nn.Conv2d(2*16, 128, 7, padding=3),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, 3, padding=1),
        #     nn.ReLU(),
        # )

        self.out = nn.Linear(latent_dim, kernel_size*kernel_size, bias=True)
        

        # alibi_slope = get_alibi_slope(n_heads // 2)
        # grid_kernel = get_grid(kernel_size, kernel_size, normalize=False).reshape(kernel_size, kernel_size, 2)
        # grid_kernel = grid_kernel - (kernel_size - 1) / 2
        # grid_kernel = -torch.abs(grid_kernel)
        # alibi_bias = torch.cat([
        #     alibi_slope.view(-1,1,1) * grid_kernel[..., 0].view(1,kernel_size,kernel_size),
        #     alibi_slope.view(-1,1,1) * grid_kernel[..., 1].view(1,kernel_size,kernel_size)
        # ]) # n_heads, kernel_size, kernel_size

        # self.register_buffer("alibi_bias", alibi_bias)

        self._init_weights()

    def _init_weights(self):
        self.out.weight.data.fill_(0)
        self.out.bias.data.fill_(0)

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
        # feat_map_up = self.attnup_feat_cnn(feat_map) # learnable upsample by 4
        # if self.upsample_factor != 4:
        #     additional_scale = float(self.upsample_factor / 4)
        #     if additional_scale > 1:
        #         feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='bilinear', align_corners=False)
        #     else:
        #         feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='nearest')

        feat_map_down = feat_map

        pad_val = (self.kernel_size - 1) // 2

        # attn_bias = self.alibi_bias.view(1,self.n_heads,self.kernel_size**2,1,1).repeat(B,1,1,H_down*self.upsample_factor,W_down*self.upsample_factor)
        # attn_bias = rearrange(attn_bias, "b k n h w -> (b h w) k 1 n")

        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.unfold(context, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        context = rearrange(context, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        context = rearrange(context, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        x = rearrange(x, 'b c h w -> (b h w) 1 c')


        # assert flow_map.shape[-2:] == feat_map.shape[-2:]


        # flow_map = rearrange(flow_map, 'b t c h w -> b (t c) h w')
        # flow_map = self.flow_mlp(flow_map)

        # nn_flow_map = F.unfold(flow_map, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        # nn_flow_map = rearrange(nn_flow_map, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        # nn_flow_map = F.interpolate(nn_flow_map, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        # nn_flow_map = rearrange(nn_flow_map, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        # up_flow_map = F.interpolate(flow_map, scale_factor=self.upsample_factor, mode="nearest") # NN up # b 2 h w
        # up_flow_map = rearrange(up_flow_map, 'b c h w -> (b h w) 1 c')

        # context = torch.cat([context, nn_flow_map], dim=-1)
        # x = torch.cat([x, up_flow_map], dim=-1)

        for lvl in range(len(self.cross_blocks)):
            # x = self.cross_blocks[lvl](x, context, attn_bias)
            x = self.cross_blocks[lvl](x, context)

        mask_out = self.out(x)
        mask_out = F.softmax(mask_out, dim=-1)
        mask_out = rearrange(mask_out, '(b h w) 1 c -> b c h w', h=H_up, w=W_up)

        return mask_out


class UpsampleTransformerV3(nn.Module):
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

        # self.attnup_feat_cnn = DecoderUpsampler(
        #     in_channels=self.latent_dim,
        #     middle_channels=self.latent_dim,
        #     out_channels=self.latent_dim,
        #     # stride=self.upsample_factor
        # )

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

        self.out = nn.Linear(latent_dim, kernel_size*kernel_size, bias=True)
        

        self._init_weights()

    def _init_weights(self):
        self.out.weight.data.fill_(0)
        self.out.bias.data.fill_(0)

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

        pad_val = (self.kernel_size - 1) // 2


        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.unfold(context, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        context = rearrange(context, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        context = rearrange(context, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        x = rearrange(x, 'b c h w -> (b h w) 1 c')

        for lvl in range(len(self.cross_blocks)):
            x = self.cross_blocks[lvl](x, context)

        mask_out = self.out(x)
        mask_out = F.softmax(mask_out, dim=-1)
        mask_out = rearrange(mask_out, '(b h w) 1 c -> b c h w', h=H_up, w=W_up)

        return mask_out


class UpsampleTransformerV4(nn.Module):
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

        # self.attnup_feat_cnn = DecoderUpsampler(
        #     in_channels=self.latent_dim,
        #     middle_channels=self.latent_dim,
        #     out_channels=self.latent_dim,
        #     # stride=self.upsample_factor
        # )

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
        self.out = Mlp(
            in_features=3*(latent_dim),
            hidden_features=3*(latent_dim),
            out_features=1
        )

        # self.out = nn.Linear(latent_dim, kernel_size*kernel_size, bias=True)
        

        self._init_weights()

    def _init_weights(self):
        pass
        # self.out.fc2.weight.data.fill_(0)
        # self.out.fc2.bias.data.fill_(0)

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
        # feat_map_up = self.attnup_feat_cnn(feat_map) # learnable upsample by 4
        # if self.upsample_factor != 4:
        #     additional_scale = float(self.upsample_factor / 4)
        #     if additional_scale > 1:
        #         feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='bilinear', align_corners=False)
        #     else:
        #         feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='nearest')

        # NOTE debugging
        coord_xy_down = get_grid(H_down, W_down, device=feat_map.device).reshape(1, H_down, W_down, 2).repeat(B, 1, 1, 1) * upsample_factor
        coord_xy_up = get_grid(H_up, W_up, device=feat_map.device).reshape(1, H_up, W_up, 2).repeat(B, 1, 1, 1)

        coord_xy_down = rearrange(coord_xy_down, 'b h w c -> b c h w')
        coord_xy_up = rearrange(coord_xy_up, 'b h w c -> b c h w')

        coord_xy_down = F.pad(coord_xy_down, (0, 1, 0, 1), "replicate")
        coord_xy_down = F.unfold(coord_xy_down, kernel_size=self.kernel_size) # B C*kernel**2 H W
        coord_xy_down = rearrange(coord_xy_down, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        coord_xy_down = F.interpolate(coord_xy_down, scale_factor=upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        coord_xy_down = rearrange(coord_xy_down, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        coord_xy_up = rearrange(coord_xy_up, 'b c h w -> (b h w) 1 c')

        coord_diff = -torch.abs(coord_xy_up - coord_xy_down).sum(-1)

        # coord_diff = rearrange(coord_diff, 'b k c -> b 1 k c') # 
        coord_diff = coord_diff.unsqueeze(1)

        alibi_slope = get_alibi_slope(self.n_heads).to(feat_map.device)
        # breakpoint()
        alibi_bias = alibi_slope.view(1,-1,1) * coord_diff
        # torch.cat([
        #     alibi_slope.view(1,-1,1) * coord_diff[..., 0],
        #     alibi_slope.view(1,-1,1) * coord_diff[..., 1]
        # ], dim=1) # (b h w) n_head k 
        alibi_bias = alibi_bias.unsqueeze(2)
        # breakpoint()
        # print(alibi_bias.shape)
        # feat_map_down_pseudo = feat_map_down_pseudo[None,None].repeat(B, feat_map.shape[1], 1, 1)

        # feat_map_up_pseudo = torch.arange(H_up*W_up).to(feat_map).reshape(H_up, W_up)
        # feat_map_up_pseudo = feat_map_up_pseudo[None,None].repeat(B, feat_map.shape[1], 1, 1)

        # feat_map = feat_map_down_pseudo
        # feat_map_up = feat_map_up_pseudo
        ####################

        feat_map_down = feat_map

        # pad_val = (self.kernel_size - 1) // 2
        # pad_val = self.kernel_size // 2

        # attn_bias = self.alibi_bias.view(1,self.n_heads,self.kernel_size**2,1,1).repeat(B,1,1,H_down*self.upsample_factor,W_down*self.upsample_factor)
        # attn_bias = rearrange(attn_bias, "b k n h w -> (b h w) k 1 n")

        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.pad(context, (0, 1, 0, 1), "replicate")

        context = F.unfold(context, kernel_size=self.kernel_size) # B C*kernel**2 H W
        context = rearrange(context, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4


        # NOTE for debug
        # context = rearrange(context, 'b (c i j) h w -> b h w (i j) c', i=self.kernel_size, j=self.kernel_size)
        # query = rearrange(feat_map_up, 'b c h w -> b h w c')
        #################
        # breakpoint()
        context = rearrange(context, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        x = rearrange(x, 'b c h w -> (b h w) 1 c')

        for lvl in range(len(self.cross_blocks)):
            x = self.cross_blocks[lvl](x, context, attn_bias=alibi_bias)

        x = self.final_norm(x)
        context = self.final_context_norm(context)
        final_input = torch.cat([
            x.repeat(1,self.kernel_size**2,1),
            context,
            x - context
        ], dim=-1) # (b h w) (i j) c

        final_output = self.out(final_input) 
        final_output = rearrange(final_output, '(b h w) c 1 -> b c h w', h=H_up, w=W_up)
        mask_out = F.softmax(final_output, dim=1)

        # mask_out = self.out(x)
        # mask_out = F.softmax(mask_out, dim=-1)
        # mask_out = rearrange(mask_out, '(b h w) 1 c -> b c h w', h=H_up, w=W_up)

        return mask_out


class UpsampleTransformerV5(nn.Module):
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
    
class UpsampleCNN(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 8,
        latent_dim: int = 128,
        hiddent_dim: Optional[int] = None,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim

        # self.n_heads = n_heads

        self.hiddent_dim = hiddent_dim if hiddent_dim is not None else latent_dim

        # breakpoint()
        self.attnup_feat_cnn = DecoderUpsampler(
            in_channels=self.latent_dim, 
            middle_channels=self.hiddent_dim, 
            out_channels=self.hiddent_dim,
            stride=self.stride,
        )

        # self.flow_mlp = nn.Sequential(
        #     nn.Conv2d(2*16, 128, 7, padding=3),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, 3, padding=1),
        #     nn.ReLU(),
        # )

        self.out = nn.Conv2d(self.hiddent_dim, kernel_size * kernel_size, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self._init_weights()

    def _init_weights(self):
        self.out.weight.data.fill_(0)
        self.out.bias.data.fill_(0)

    def forward(
        self,
        feat_map: Float[Tensor, "b c1 h w"],
        # flow_map: Float[Tensor, "b c2 h w"],
    ):

        # assert flow_map.shape[-2:] == feat_map.shape[-2:]

        feat_map = self.attnup_feat_cnn(feat_map)  # learnable upsample by 4

        mask_out = self.out(feat_map) # B C H W
        mask_out = F.softmax(mask_out, dim=1)

        return mask_out
    
class UpsampleCNNV2(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 8,
        latent_dim: int = 128,
        hiddent_dim: Optional[int] = None,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim

        # self.n_heads = n_heads

        self.hiddent_dim = hiddent_dim if hiddent_dim is not None else latent_dim

        # breakpoint()
        self.attnup_feat_cnn = DecoderUpsampler(
            in_channels=self.latent_dim, 
            middle_channels=self.hiddent_dim, 
            out_channels=self.hiddent_dim,
            stride=self.stride,
        )

        # self.flow_mlp = nn.Sequential(
        #     nn.Conv2d(2*16, 128, 7, padding=3),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, 3, padding=1),
        #     nn.ReLU(),
        # )

        self.out = nn.Conv2d(self.hiddent_dim, kernel_size * kernel_size, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self._init_weights()

    def _init_weights(self):
        self.out.weight.data.fill_(0)
        self.out.bias.data.fill_(0)

    def forward(
        self,
        feat_map: Float[Tensor, "b c1 h w"],
        # flow_map: Float[Tensor, "b c2 h w"],
    ):

        # assert flow_map.shape[-2:] == feat_map.shape[-2:]

        feat_map = self.attnup_feat_cnn(feat_map)  # learnable upsample by 4

        mask_out = self.out(feat_map) # B C H W
        mask_out = F.softmax(mask_out, dim=1)

        return mask_out

class UpsampleCNNV2Temporal(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 8,
        latent_dim: int = 128,
        hiddent_dim: Optional[int] = None,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim

        # self.n_heads = n_heads

        self.hiddent_dim = hiddent_dim if hiddent_dim is not None else latent_dim

        # breakpoint()
        self.attnup_feat_cnn = DecoderUpsampler(
            in_channels=self.latent_dim, 
            middle_channels=self.hiddent_dim, 
            out_channels=self.hiddent_dim,
            stride=self.stride,
        )

        self.out = nn.Conv2d(self.hiddent_dim, kernel_size * kernel_size, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    self.hiddent_dim,
                    4,
                    mlp_ratio=4,
                    attn_class=Attention,
                    flash=False,
                    dim_head=32,
                )
                for _ in range(2)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hiddent_dim, elementwise_affine=False, eps=1e-6)

            # self.first_block_conv = Conv1dPad(
            #     in_channels=self.hiddent_dim, 
            #     out_channels=self.hiddent_dim, 
            #     kernel_size=3, 
            #     stride=1
            # )
            # # self.first_block_norm = nn.InstanceNorm1d(self.hiddent_dim)
            # self.first_block_relu = nn.ReLU()

            # self.time_blocks = []
            # for i in range(3):
            #     self.time_blocks.append(
            #         ResidualBlock1d(
            #             self.hiddent_dim,
            #             self.hiddent_dim,
            #             kernel_size=3,
            #             stride=1,
            #             groups=1,
            #             use_norm=True,
            #             use_do=False,
            #             is_first_block=(i==0)
            #         )
            #     )
            # self.time_blocks = nn.ModuleList(self.time_blocks)
            # self.final_relu = nn.ReLU()
        


        self._init_weights()

    def _init_weights(self):
        self.out.weight.data.fill_(0)
        self.out.bias.data.fill_(0)

    def forward(
        self,
        feat_map: Float[Tensor, "b t c1 h w"],
        # flow_map: Float[Tensor, "b c2 h w"],
    ):

        # assert flow_map.shape[-2:] == feat_map.shape[-2:]
        B, T = feat_map.shape[:2]

        feat_map = rearrange(feat_map, 'b t c h w -> (b t) c h w')
        feat_map = self.attnup_feat_cnn(feat_map)  # learnable upsample by 4

        h_up, w_up = feat_map.shape[-2:]

        # NOTE attn based
        feat_map = rearrange(feat_map, '(b t) c h w -> (b h w) t c', b=B, t=T)
        for lvl in range(len(self.time_blocks)):
            # print(feat_map.shape)
            feat_map = self.time_blocks[lvl](feat_map)
        feat_map = self.final_norm(feat_map)
        feat_map = rearrange(feat_map, '(b h w) t c -> (b t) c h w', b=B, h=h_up, w=w_up)

            # # NOTE conv based
            # feat_map = rearrange(feat_map, '(b t) c h w -> (b h w) c t', b=B, t=T)
            # feat_map = self.first_block_conv(feat_map)
            # feat_map = self.first_block_relu(feat_map)
            # for t_block in self.time_blocks:
            #     feat_map = t_block(feat_map)
            # feat_map = self.final_relu(feat_map)
            # feat_map = rearrange(feat_map, '(b h w) c t -> (b t) c h w', b=B, h=h_up, w=w_up)
            # #####################################

        mask_out = self.out(feat_map) # B C H W
        mask_out = F.softmax(mask_out, dim=1)

        mask_out = rearrange(mask_out, '(b t) c h w -> b t c h w', b=B, t=T)  

        return mask_out
