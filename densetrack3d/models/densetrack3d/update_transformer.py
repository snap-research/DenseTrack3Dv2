import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from densetrack3d.models.densetrack3d.blocks import Attention, AttnBlock, CrossAttnBlock, Mlp
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int64, Shaped
from torch import Tensor, nn


class EfficientUpdateFormerV2(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        num_blocks=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
        flash=False,
        use_local_attn=True,
        linear_layer_for_vis_conf=False,
        linear_layer_for_feat=False,
        feat_dim=-1
    ):
        super().__init__()

        self.num_blocks = num_blocks
        # self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.use_local_attn = use_local_attn
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.linear_layer_for_feat = linear_layer_for_feat

        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)
        # self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)
        if linear_layer_for_vis_conf:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)

        if linear_layer_for_feat:
            assert feat_dim > 0
            self.track_feat_projector = nn.Linear(hidden_size, feat_dim, bias=True)
            self.norm = nn.GroupNorm(1, feat_dim)
            self.track_feat_updater = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
            )

        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                    flash=flash,
                )
                for _ in range(num_blocks)
            ]
        )

        self.space_virtual_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                    flash=flash,
                )
                for _ in range(num_blocks)
            ]
        )
        self.space_point2virtual_blocks = nn.ModuleList(
            [
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash)
                for _ in range(num_blocks)
            ]
        )
        self.space_virtual2point_blocks = nn.ModuleList(
            [
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash)
                for _ in range(num_blocks)
            ]
        )

        if self.use_local_attn:
            self.space_local_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention,
                        flash=flash,
                    )
                    for _ in range(num_blocks)
                ]
            )

        self.local_size = 6

        # assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        # def _basic_init(module):
        #     if isinstance(module, nn.Linear):
        #         torch.nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            if self.linear_layer_for_vis_conf:
                torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        self.apply(_basic_init)

        if self.use_local_attn:
            # NOTE zero-init local attention
            for i in range(len(self.space_local_blocks)):
                self.space_local_blocks[i].mlp.fc2.weight.data.fill_(0.0)
                self.space_local_blocks[i].mlp.fc2.bias.data.fill_(0.0)
                self.space_local_blocks[i].attn.to_out.weight.data.fill_(0.0)
                self.space_local_blocks[i].attn.to_out.bias.data.fill_(0.0)

    def local_attention(self, point_tokens, dH, dW, j, B, T):
        # NOTE local attention
        local_size = self.local_size
        shift_size = local_size // 2

        orig_dH, orig_dW = dH, dW

        local_patches = rearrange(point_tokens, "b (h w) c -> b c h w", h=dH, w=dW)
        pad_h = local_size - local_patches.shape[-2] % local_size if local_patches.shape[-2] % local_size != 0 else 0
        pad_w = local_size - local_patches.shape[-1] % local_size if local_patches.shape[-1] % local_size != 0 else 0

        if pad_h > 0 or pad_w > 0:
            local_patches = F.pad(local_patches, (0, pad_w, 0, pad_h), "constant", 0)
            dH, dW = local_patches.shape[-2], local_patches.shape[-1]
        # if i % 2 == 1:
        #     use_shift = True
        # local_patches = torch.roll(local_patches, )
        # copy_local_patches = local_patches.clone()
        local_patches = F.unfold(local_patches, kernel_size=local_size, stride=local_size)  # (B T) C (H W)
        local_patches = rearrange(local_patches, "b (c p1 p2) l -> (b l) (p1 p2) c", p1=local_size, p2=local_size)

        attn_mask = local_patches.detach().abs().sum(-1) > 0  # B N

        # NOTE add embedding here
        # local_embed = self.space_local_emb.unsqueeze(0).repeat(local_patches.shape[0], 1, 1)
        # local_patches = local_patches + local_embed
        # breakpoint()
        local_patches = self.space_local_blocks[j](local_patches, mask=attn_mask)

        # breakpoint()
        local_patches = rearrange(
            local_patches,
            "(b h w) (p1 p2) c -> b c h p1 w p2",
            b=B * T,
            h=dH // local_size,
            w=dW // local_size,
            p1=local_size,
            p2=local_size,
        )

        local_patches = local_patches.contiguous().view(B * T, -1, dH, dW)

        local_patches = local_patches[:, :, :orig_dH, :orig_dW]
        point_tokens = rearrange(local_patches, "b c h w -> b (h w) c")

        return point_tokens

    def forward(
        self,
        input_tensor: Float[Tensor, "b t n c_in"],
        attn_mask: Bool[Tensor, "b t n"] = None,
        n_sparse: int = 256,
        dH: int = 4,
        dW: int = 4,
        use_efficient_global_attn: bool = False,
        use_local_attn: bool = True,
    ) -> Float[Tensor, "b t n c_out"]:

        if use_efficient_global_attn:
            assert n_sparse > 0

        B, T, *_ = input_tensor.shape

        real_tokens = self.input_transform(input_tensor)
        virtual_tokens = repeat(self.virual_tracks, "1 n 1 c -> b n t c", b=B, t=T)
        virtual_tokens = rearrange(virtual_tokens, "b n t c -> b t n c")

        # self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([real_tokens, virtual_tokens], dim=2)
        N_total = tokens.shape[2]

        # j = 0

        tokens = rearrange(tokens, "b t n c -> (b n) t c")
        attn_mask = rearrange(attn_mask, "b t n -> (b t) n")

        for lvl in range(self.num_blocks):
            # tokens = rearrange(tokens, 'b t n c -> (b n) t c')
            tokens = self.time_blocks[lvl](tokens)

            tokens = rearrange(tokens, "(b n) t c -> (b t) n c", b=B, t=T)

            virtual_tokens = tokens[:, N_total - self.num_virtual_tracks :]
            real_tokens = tokens[:, : N_total - self.num_virtual_tracks]
            sparse_tokens = real_tokens[:, :n_sparse]
            dense_tokens = real_tokens[:, n_sparse:]

            # NOTE global attention

            if use_efficient_global_attn:
                sparse_mask = attn_mask[:, :n_sparse]
                virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, sparse_tokens, mask=sparse_mask)
            else:
                virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, real_tokens, mask=attn_mask)

            virtual_tokens = self.space_virtual_blocks[lvl](virtual_tokens)

            # NOTE local attention
            if self.use_local_attn and use_local_attn and dH > 0 and dW > 0:
                dense_tokens = self.local_attention(dense_tokens, dH, dW, lvl, B, T)
                real_tokens = torch.cat([sparse_tokens, dense_tokens], dim=1)

            real_tokens = self.space_point2virtual_blocks[lvl](real_tokens, virtual_tokens, mask=attn_mask)

            tokens = torch.cat([real_tokens, virtual_tokens], dim=1)

            if lvl == self.num_blocks - 1:  # NOTE last layer: no virtual tokens
                tokens = rearrange(tokens, "(b t) n c -> b t n c", b=B, t=T)
            else:
                tokens = rearrange(tokens, "(b t) n c -> (b n) t c", b=B, t=T)
            # tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  # (B T) N C -> B N T C

        # tokens = rearrange(tokens, '(b n) t c -> b t n c',)

        real_tokens = tokens[:, :, : N_total - self.num_virtual_tracks]
        flow = self.flow_head(real_tokens)

        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(real_tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        if self.linear_layer_for_feat:
            delta_feat = self.track_feat_projector(real_tokens)
            delta_feat = self.track_feat_updater(self.norm(rearrange(delta_feat, "b t n c -> (b t n) c")))
            delta_feat = rearrange(delta_feat, "(b t n) c -> b t n c", b=B, t=T) 

            flow = torch.cat([flow, delta_feat], dim=-1)

        return flow

class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        num_blocks=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
        flash=False,
        use_local_attn=True,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        # self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.use_local_attn = use_local_attn

        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                    flash=flash,
                )
                for _ in range(num_blocks)
            ]
        )

        if self.add_space_attn:
            self.num_virtual_tracks = num_virtual_tracks
            self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))
        

            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention,
                        flash=flash,
                    )
                    for _ in range(num_blocks)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash)
                    for _ in range(num_blocks)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash)
                    for _ in range(num_blocks)
                ]
            )

            if self.use_local_attn:
                self.space_local_blocks = nn.ModuleList(
                    [
                        AttnBlock(
                            hidden_size,
                            num_heads,
                            mlp_ratio=mlp_ratio,
                            attn_class=Attention,
                            flash=flash,
                        )
                        for _ in range(num_blocks)
                    ]
                )

            self.local_size = 6

        # assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def local_attention(self, point_tokens, dH, dW, j, B, T):
        # NOTE local attention
        local_size = self.local_size
        shift_size = local_size // 2

        orig_dH, orig_dW = dH, dW

        local_patches = rearrange(point_tokens, "b (h w) c -> b c h w", h=dH, w=dW)
        pad_h = local_size - local_patches.shape[-2] % local_size if local_patches.shape[-2] % local_size != 0 else 0
        pad_w = local_size - local_patches.shape[-1] % local_size if local_patches.shape[-1] % local_size != 0 else 0

        if pad_h > 0 or pad_w > 0:
            local_patches = F.pad(local_patches, (0, pad_w, 0, pad_h), "constant", 0)
            dH, dW = local_patches.shape[-2], local_patches.shape[-1]
        # if i % 2 == 1:
        #     use_shift = True
        # local_patches = torch.roll(local_patches, )
        # copy_local_patches = local_patches.clone()
        local_patches = F.unfold(local_patches, kernel_size=local_size, stride=local_size)  # (B T) C (H W)
        local_patches = rearrange(local_patches, "b (c p1 p2) l -> (b l) (p1 p2) c", p1=local_size, p2=local_size)

        attn_mask = local_patches.detach().abs().sum(-1) > 0  # B N

        # NOTE add embedding here
        # local_embed = self.space_local_emb.unsqueeze(0).repeat(local_patches.shape[0], 1, 1)
        # local_patches = local_patches + local_embed
        # breakpoint()
        local_patches = self.space_local_blocks[j](local_patches, mask=attn_mask)

        # breakpoint()
        local_patches = rearrange(
            local_patches,
            "(b h w) (p1 p2) c -> b c h p1 w p2",
            b=B * T,
            h=dH // local_size,
            w=dW // local_size,
            p1=local_size,
            p2=local_size,
        )

        # breakpoint()
        local_patches = local_patches.contiguous().view(B * T, -1, dH, dW)

        local_patches = local_patches[:, :, :orig_dH, :orig_dW]
        point_tokens = rearrange(local_patches, "b c h w -> b (h w) c")

        return point_tokens

    def forward(
        self,
        input_tensor: Float[Tensor, "b t n c_in"],
        attn_mask: Bool[Tensor, "b t n"] = None,
        n_sparse: int = 256,
        dH: int = 4,
        dW: int = 4,
        use_efficient_global_attn: bool = False,
        use_local_attn: bool = True,
    ) -> Float[Tensor, "b t n c_out"]:

        if use_efficient_global_attn:
            assert n_sparse > 0

        B, T, *_ = input_tensor.shape

        real_tokens = self.input_transform(input_tensor)

        if self.add_space_attn:
            virtual_tokens = repeat(self.virual_tracks, "1 n 1 c -> b n t c", b=B, t=T)
            virtual_tokens = rearrange(virtual_tokens, "b n t c -> b t n c")

            # self.virual_tracks.repeat(B, 1, T, 1)
            tokens = torch.cat([real_tokens, virtual_tokens], dim=2)
            N_total = tokens.shape[2]
        else:
            tokens = real_tokens
            N_total = tokens.shape[2]

        # j = 0

        tokens = rearrange(tokens, "b t n c -> (b n) t c")
        attn_mask = rearrange(attn_mask, "b t n -> (b t) n")

        for lvl in range(self.num_blocks):
            # tokens = rearrange(tokens, 'b t n c -> (b n) t c')
            tokens = self.time_blocks[lvl](tokens)

            if self.add_space_attn:
                tokens = rearrange(tokens, "(b n) t c -> (b t) n c", b=B, t=T)

                virtual_tokens = tokens[:, N_total - self.num_virtual_tracks :]
                real_tokens = tokens[:, : N_total - self.num_virtual_tracks]
                sparse_tokens = real_tokens[:, :n_sparse]
                dense_tokens = real_tokens[:, n_sparse:]

                # NOTE global attention

                if use_efficient_global_attn:
                    sparse_mask = attn_mask[:, :n_sparse]
                    virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, sparse_tokens, mask=sparse_mask)
                else:
                    virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, real_tokens, mask=attn_mask)

                virtual_tokens = self.space_virtual_blocks[lvl](virtual_tokens)

                # NOTE local attention
                if self.use_local_attn and use_local_attn and dH > 0 and dW > 0:
                    dense_tokens = self.local_attention(dense_tokens, dH, dW, lvl, B, T)
                    real_tokens = torch.cat([sparse_tokens, dense_tokens], dim=1)

                real_tokens = self.space_point2virtual_blocks[lvl](real_tokens, virtual_tokens, mask=attn_mask)

                tokens = torch.cat([real_tokens, virtual_tokens], dim=1)

                if lvl == self.num_blocks - 1:  # NOTE last layer: no virtual tokens
                    tokens = rearrange(tokens, "(b t) n c -> b t n c", b=B, t=T)
                else:
                    tokens = rearrange(tokens, "(b t) n c -> (b n) t c", b=B, t=T)

            else:
                if lvl == self.num_blocks - 1:  # NOTE last layer: no virtual tokens
                    tokens = rearrange(tokens, "(b n) t c -> b t n c", b=B)
            # tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  # (B T) N C -> B N T C

        # tokens = rearrange(tokens, '(b n) t c -> b t n c',)
        if self.add_space_attn:
            real_tokens = tokens[:, :, : N_total - self.num_virtual_tracks]
            flow = self.flow_head(real_tokens)
        else:
            flow = self.flow_head(tokens)
        return flow

    def forward_ablate(
        self,
        input_tensor: Float[Tensor, "b t n c_in"],
        attn_mask: Bool[Tensor, "b t n"] = None,
        n_sparse: int = 256,
        dH: int = 4,
        dW: int = 4,
        use_efficient_global_attn: bool = False,
        use_local_attn: bool = True,
        skip_spatial: bool = False,
    ) -> Float[Tensor, "b t n c_out"]:

        if use_efficient_global_attn:
            assert n_sparse > 0

        B, T, *_ = input_tensor.shape

        real_tokens = self.input_transform(input_tensor)
        virtual_tokens = repeat(self.virual_tracks, "1 n 1 c -> b n t c", b=B, t=T)
        virtual_tokens = rearrange(virtual_tokens, "b n t c -> b t n c")

        # self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([real_tokens, virtual_tokens], dim=2)
        N_total = tokens.shape[2]

        # j = 0

        tokens = rearrange(tokens, "b t n c -> (b n) t c")
        attn_mask = rearrange(attn_mask, "b t n -> (b t) n")

        for lvl in range(self.num_blocks):
            # tokens = rearrange(tokens, 'b t n c -> (b n) t c')
            tokens = self.time_blocks[lvl](tokens)

            tokens = rearrange(tokens, "(b n) t c -> (b t) n c", b=B, t=T)

            if not skip_spatial:
                virtual_tokens = tokens[:, N_total - self.num_virtual_tracks :]
                real_tokens = tokens[:, : N_total - self.num_virtual_tracks]
                sparse_tokens = real_tokens[:, :n_sparse]
                dense_tokens = real_tokens[:, n_sparse:]

                # NOTE global attention

                if use_efficient_global_attn:
                    sparse_mask = attn_mask[:, :n_sparse]
                    virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, sparse_tokens, mask=sparse_mask)
                else:
                    virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, real_tokens, mask=attn_mask)

                virtual_tokens = self.space_virtual_blocks[lvl](virtual_tokens)

                # NOTE local attention
                if self.use_local_attn and use_local_attn and dH > 0 and dW > 0:
                    dense_tokens = self.local_attention(dense_tokens, dH, dW, lvl, B, T)
                    real_tokens = torch.cat([sparse_tokens, dense_tokens], dim=1)

                real_tokens = self.space_point2virtual_blocks[lvl](real_tokens, virtual_tokens, mask=attn_mask)

                tokens = torch.cat([real_tokens, virtual_tokens], dim=1)

            else:
                # virtual_tokens = tokens[:, N_total - self.num_virtual_tracks :]
                dense_tokens = tokens[:, n_sparse: N_total - self.num_virtual_tracks]

                # breakpoint()
                dense_tokens = self.local_attention(dense_tokens, dH, dW, lvl, B, T)
                tokens = torch.cat([tokens[:, :n_sparse], dense_tokens, tokens[:, N_total - self.num_virtual_tracks:]], dim=1)

            if lvl == self.num_blocks - 1:  # NOTE last layer: no virtual tokens
                tokens = rearrange(tokens, "(b t) n c -> b t n c", b=B, t=T)
            else:
                tokens = rearrange(tokens, "(b t) n c -> (b n) t c", b=B, t=T)
            # tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  # (B T) N C -> B N T C

        # tokens = rearrange(tokens, '(b n) t c -> b t n c',)

        real_tokens = tokens[:, :, : N_total - self.num_virtual_tracks]
        flow = self.flow_head(real_tokens)
        return flow

    def forward_simple(
        self,
        input_tensor: Float[Tensor, "b t n c_in"],
        attn_mask: Bool[Tensor, "b t n"] = None,
        n_sparse: int = 256,
        dH: int = 4,
        dW: int = 4,
        use_efficient_global_attn: bool = False,
        use_local_attn: bool = True,
    ) -> Float[Tensor, "b t n c_out"]:

        if use_efficient_global_attn:
            assert n_sparse > 0

        B, T, *_ = input_tensor.shape

        tokens = self.input_transform(input_tensor)
        # virtual_tokens = repeat(self.virual_tracks, "1 n 1 c -> b n t c", b=B, t=T)
        # virtual_tokens = rearrange(virtual_tokens, "b n t c -> b t n c")

        # self.virual_tracks.repeat(B, 1, T, 1)
        # tokens = torch.cat([real_tokens, virtual_tokens], dim=2)
        N_total = tokens.shape[2]

        # j = 0

        tokens = rearrange(tokens, "b t n c -> (b n) t c")
        attn_mask = rearrange(attn_mask, "b t n -> (b t) n")

        for lvl in range(self.num_blocks):
            # tokens = rearrange(tokens, 'b t n c -> (b n) t c')
            tokens = self.time_blocks[lvl](tokens)

            tokens = rearrange(tokens, "(b n) t c -> (b t) n c", b=B, t=T)

            tokens = self.space_virtual_blocks[lvl](tokens)

            # virtual_tokens = tokens[:, N_total - self.num_virtual_tracks :]
            # real_tokens = tokens[:, : N_total - self.num_virtual_tracks]
            # sparse_tokens = real_tokens[:, :n_sparse]
            # dense_tokens = real_tokens[:, n_sparse:]

            # # NOTE global attention

            # if use_efficient_global_attn:
            #     sparse_mask = attn_mask[:, :n_sparse]
            #     virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, sparse_tokens, mask=sparse_mask)
            # else:
            #     virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, real_tokens, mask=attn_mask)

            # virtual_tokens = self.space_virtual_blocks[lvl](virtual_tokens)

            # # NOTE local attention
            # if self.use_local_attn and use_local_attn and dH > 0 and dW > 0:
            #     dense_tokens = self.local_attention(dense_tokens, dH, dW, lvl, B, T)
            #     real_tokens = torch.cat([sparse_tokens, dense_tokens], dim=1)

            # real_tokens = self.space_point2virtual_blocks[lvl](real_tokens, virtual_tokens, mask=attn_mask)

            # tokens = torch.cat([real_tokens, virtual_tokens], dim=1)

            if lvl == self.num_blocks - 1:  # NOTE last layer: no virtual tokens
                tokens = rearrange(tokens, "(b t) n c -> b t n c", b=B, t=T)
            else:
                tokens = rearrange(tokens, "(b t) n c -> (b n) t c", b=B, t=T)
            # tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  # (B T) N C -> B N T C

        # tokens = rearrange(tokens, '(b n) t c -> b t n c',)

        # real_tokens = tokens[:, :, : N_total - self.num_virtual_tracks]
        flow = self.flow_head(tokens)
        return flow

class EfficientUpdateFormer2(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        num_blocks=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
        flash=False,
        use_local_attn=True,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        # self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.use_local_attn = use_local_attn

        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                    flash=flash,
                )
                for _ in range(num_blocks)
            ]
        )

        self.space_virtual_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                    flash=flash,
                )
                for _ in range(num_blocks)
            ]
        )
        self.space_point2virtual_blocks = nn.ModuleList(
            [
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash)
                for _ in range(num_blocks)
            ]
        )
        self.space_virtual2point_blocks = nn.ModuleList(
            [
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash)
                for _ in range(num_blocks)
            ]
        )

        # if self.use_local_attn:
        #     self.space_local_blocks = nn.ModuleList(
        #         [
        #             AttnBlock(
        #                 hidden_size,
        #                 num_heads,
        #                 mlp_ratio=mlp_ratio,
        #                 attn_class=Attention,
        #                 flash=flash,
        #             )
        #             for _ in range(num_blocks)
        #         ]
        #     )

        # self.local_size = 6

        # assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(
        self,
        input_tensor: Float[Tensor, "b t n c_in"],
        attn_mask: Bool[Tensor, "b t n"] = None,
        n_sparse: int = 256,
        dH: int = 4,
        dW: int = 4,
        use_efficient_global_attn: bool = False,
        use_local_attn: bool = True,
        skip_dense_spatial_attn: bool = False,
    ) -> Float[Tensor, "b t n c_out"]:

        if use_efficient_global_attn:
            assert n_sparse > 0

        B, T, *_ = input_tensor.shape

        real_tokens = self.input_transform(input_tensor)
        virtual_tokens = repeat(self.virual_tracks, "1 n 1 c -> b n t c", b=B, t=T)
        virtual_tokens = rearrange(virtual_tokens, "b n t c -> b t n c")

        tokens = torch.cat([real_tokens, virtual_tokens], dim=2)
        N_total = tokens.shape[2]

        tokens = rearrange(tokens, "b t n c -> (b n) t c")
        attn_mask = rearrange(attn_mask, "b t n -> (b t) n")

        for lvl in range(self.num_blocks):

            tokens = self.time_blocks[lvl](tokens)
            tokens = rearrange(tokens, "(b n) t c -> (b t) n c", b=B, t=T)

            virtual_tokens = tokens[:, N_total - self.num_virtual_tracks :]
            real_tokens = tokens[:, : N_total - self.num_virtual_tracks]
            sparse_tokens = real_tokens[:, :n_sparse]
            dense_tokens = real_tokens[:, n_sparse:]
            
            if skip_dense_spatial_attn:
                if n_sparse > 0:
                    sparse_mask = attn_mask[:, :n_sparse]
                    virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, sparse_tokens, mask=sparse_mask)
                    virtual_tokens = self.space_virtual_blocks[lvl](virtual_tokens)
                    sparse_tokens = self.space_point2virtual_blocks[lvl](sparse_tokens, virtual_tokens, mask=sparse_mask)

                    tokens = torch.cat([sparse_tokens, dense_tokens, virtual_tokens], dim=1)
                    
            else:

                # NOTE global attention
                if use_efficient_global_attn:
                    sparse_mask = attn_mask[:, :n_sparse]
                    virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, sparse_tokens, mask=sparse_mask)
                else:
                    virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, real_tokens, mask=attn_mask)

                virtual_tokens = self.space_virtual_blocks[lvl](virtual_tokens)

                # NOTE local attention
                if self.use_local_attn and use_local_attn and dH > 0 and dW > 0:
                    dense_tokens = self.local_attention(dense_tokens, dH, dW, lvl, B, T)
                    real_tokens = torch.cat([sparse_tokens, dense_tokens], dim=1)

                real_tokens = self.space_point2virtual_blocks[lvl](real_tokens, virtual_tokens, mask=attn_mask)

                tokens = torch.cat([real_tokens, virtual_tokens], dim=1)

            if lvl == self.num_blocks - 1:  # NOTE last layer: no virtual tokens
                tokens = rearrange(tokens, "(b t) n c -> b t n c", b=B, t=T)
            else:
                tokens = rearrange(tokens, "(b t) n c -> (b n) t c", b=B, t=T)

        real_tokens = tokens[:, :, : N_total - self.num_virtual_tracks]
        flow = self.flow_head(real_tokens)
        return flow
    
