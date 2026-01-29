import math
import warnings
from collections.abc import Callable, Sequence
from functools import partial

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor, nn

from fm_energy.models.nn_components import (
    EmbedderWrapper,
    LearnedSinusoidalPosEmb,
    RandomSinusoidalPosEmb,
    SinusoidalPosEmb,
    default,
)


class UnetAttention(nn.Module):
    """(batch, dim, sequence) -> (batch, dim, sequence)
    difference with LinearSelfAttention:
        1. LinearSelfAttention has a RMSNorm after the output, while Attention does not.
        2. LinearSelfAttention normalizes over heads and sequence,
        while Attention normalizes over sequence only.
    """

    def __init__(self, dim=int, num_head: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.num_head = num_head
        hidden_dim = dim_head * num_head

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: Float[Tensor, ""]) -> Float[Tensor, ""]:
        b, c, _l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            rearrange(t, "b (num_head d) l -> b num_head d l", num_head=self.num_head)
            for t in qkv
        )

        q = q * self.scale
        similarity = einsum(q, k, "b h d lq, b h d lk -> b h lq lk")
        similarity = similarity.softmax(dim=-1)  # over key/value
        out = einsum(similarity, v, "b h lq lk, b h d lk -> b h lq d")
        out = rearrange(out, "b h lq d -> b (h d) lq")

        return self.to_out(out)


class RMSNorm(nn.Module):
    "do normalization over channel dimension and multiply with a learnable parameter g"

    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: Float[Tensor, ""]) -> Float[Tensor, ""]:
        return F.normalize(x, dim=1) * self.g * math.sqrt(x.shape[1])


class UnetLinearSelfAttention(nn.Module):
    r"""(batch, dim, sequence) -> (batch, dim, sequence)

    See: https://openaccess.thecvf.com/content/WACV2021/html/Shen_Efficient_Attention_Attention_With_Linear_Complexities_WACV_2021_paper.html
    """

    def __init__(self, dim: int, num_head: int = 4, dim_head: int | None = None):
        super().__init__()
        self.dim_inout = dim
        self.scale = dim_head**-0.5
        self.num_head = num_head
        self.hidden_dim = dim_head * num_head
        self.to_qkv = nn.Conv1d(self.dim_inout, self.hidden_dim * 3, 1, bias=True)

        self.to_out = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.dim_inout, 1), RMSNorm(self.dim_inout)
        )
        self.rms_norm_q = nn.Sequential(
            Rearrange("b num_head d l -> b (num_head d) l"),
            RMSNorm(self.hidden_dim),  # in dim 1
            Rearrange("b (num_head d) l -> b num_head d l", num_head=self.num_head),
        )
        self.rms_norm_k = nn.Sequential(
            Rearrange("b num_head d l -> b (num_head d) l"),
            RMSNorm(self.hidden_dim),  # in dim 1
            Rearrange("b (num_head d) l -> b num_head d l", num_head=self.num_head),
        )

    def forward(self, x: Float[Tensor, ""]) -> Float[Tensor, ""]:
        qkv = self.to_qkv(x).chunk(3, dim=1)  # 3 * (batch, hidden_dim, sequence)
        q, k, v = (
            rearrange(t, "b (num_head d) l -> b num_head d l", num_head=self.num_head)
            for t in qkv
        )  # shape: 3 * (batch, num_head, dim_head, sequence)

        # such normalization: one independent attention for each head
        q = self.rms_norm_q(q)
        k = self.rms_norm_k(k)
        # q = q.softmax(dim = -2) # over dim_head
        # k = k.softmax(dim = -1) # over sequence
        q = q * self.scale  # scale

        sim = einsum(q, k, "b h dq l, b h dq n -> b h l n")
        sim = sim.softmax(dim=-1)  # over sequence
        # out = einsum(context, q, 'b h dq dv, b h dq l -> b h dv l')
        out = einsum(
            sim, v, "b h l n, b h dv n -> b h dv l"
        )  # shape: (batch, num_head, dim_head, sequence)
        out = rearrange(
            out, "b h dv l -> b (h dv) l"
        )  # shape: (batch, hidden_dim, sequence)
        return self.to_out(out)  # shape: (batch, dim, sequence)


class ResnetSubBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_group: int | None = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(num_group, dim_out)
        self.activation = nn.SiLU()

    def forward(
        self,
        x: Float[Tensor, ""],
        scale_shift: tuple[Float[Tensor, ""], Float[Tensor, ""]] | None = None,
    ) -> Float[Tensor, ""]:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift
        x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    """
    Residual block for 1D convolutional neural networks.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        time_emb_dim (Optional[int]): Dimension of time embedding. Defaults to None.
            will be used to compute scale and shift using MLP.
            so dimension is can be different from dim_out.
        num_group (int): Number of groups for group normalization. Defaults to 8.

    Forward:
        Args:
            x (Float[Tensor, ""]): Input tensor of shape (batch, dim_in, sequence).
            time_emb (Optional[Float[Tensor, 'batch time_emb_dim']]):
            Time embedding tensor of shape (batch, time_emb_dim).
        Returns:
            Float[Tensor, ""]: Output tensor of shape (batch, dim_out, sequence).
        Process:
            1. If time_emb is given, use it to compute scale and shift.
            2. Apply two ResnetSubBlock sequentially.
            3. Add residual connection.

    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_time_emb: int | None = None,
        num_group=8,
        dropout=0.1,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(dim_time_emb, dim_out * 2))
            if dim_time_emb is not None
            else None
        )

        self.block1 = ResnetSubBlock(dim_in, dim_out, num_group)
        self.block2 = ResnetSubBlock(dim_out, dim_out, num_group)
        # residual_conv: if dim_in == dim_out, use identity,
        # else use 1x1 conv (=linear transformation)
        self.residual_conv = (
            nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        )
        self.dropout = dropout

    def forward(
        self,
        x: Float[Tensor, ""],
        time_emb: Float[Tensor, "batch dim_time_emb"] | None,
    ) -> Float[Tensor, ""]:
        # if time_emb is given, use it to compute scale and shift
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)  # shape: (batch, dim_out * 2)
            time_emb = rearrange(
                time_emb, "b c -> b c 1"
            )  # shape: (batch, dim_out * 2, 1)
            scale_shift = time_emb.chunk(
                2, dim=1
            )  # shape: (batch, dim_out, 1), (batch, dim_out, 1)

        h = self.block1(x, scale_shift)  # shape: (batch, dim_out, sequence)
        h = self.block2(h)  # shape: (batch, dim_out, sequence)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h + self.residual_conv(x)  # shape: (batch, dim_out, sequence)


class PreNorm(nn.Module):
    "add pre-normalization to given callable"

    def __init__(self, dim: int, func: Callable):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.func = func

    def forward(self, x: Float[Tensor, ""]) -> Float[Tensor, ""]:
        return self.func(self.norm(x))


class Downsample(nn.Module):
    def __init__(self, dim_in: int, dim_out: int | None = 3):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.downsample_conv = nn.Conv1d(dim_in, dim_out, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample_conv(x)
        return x


class Upsample(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int = 3,
        scale_factor: int = 2,
        mode: str = "nearest",
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.scale_factor = scale_factor
        self.mode = mode
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv = nn.Conv1d(dim_in, dim_out, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x


class Residual(nn.Module):
    def __init__(self, func: Callable[[Tensor], Tensor]):
        super().__init__()
        self.func = func

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.func(x, *args, **kwargs) + x


class Unet1D(nn.Module):
    """
    UNet1D

    processing condition: add the embedded condition to time embedding.
    """

    def __init__(
        self,
        dim_base: int,  # base dim for conv, not dim of input
        dim_out: int | None = None,  # output dim for conv, final output
        dim_mult: Sequence[int] = (1, 2, 4, 8),  # multiplier for each resolution
        num_in_channel: int = 3,
        self_condition: bool = False,
        num_resnet_block_group: int = 8,
        learn_variance: bool = False,
        type_pos_emb: str = "sinusoidal",
        dim_learned_pos_emb: int = 16,
        # dim_attn_head: int = 32,
        num_attn_head: int = 4,
        dropout: float = 0.1,
        conditioning: bool = False,
        cond_embedder: EmbedderWrapper | None = None,
    ):
        super().__init__()
        assert type_pos_emb in {
            "sinusoidal",
            "learned",
            "random",
        }, "positional embedding type must be one of sinusoidal, learned, random"
        self.type_pos_emb = type_pos_emb
        self.dim_base = dim_base  # dimensions
        self.conditioning = conditioning
        self.dropout = dropout

        # dimensions
        self.num_in_channel = num_in_channel
        self.self_condition = self_condition
        num_in_channel_init_conv = num_in_channel * (2 if self_condition else 1)

        self.init_conv = nn.Conv1d(
            num_in_channel_init_conv, dim_base, 7, padding=3
        )  # (batch, num_in_channel, sequence) -> (batch, dim_base, sequence)
        self.dim_out = default(
            dim_out, num_in_channel * (1 if not learn_variance else 2)
        )

        list_dim = [*(dim_base * t for t in dim_mult)]  # base * multipliers
        list_dim_in_out = list(
            zip(list_dim[:-1], list_dim[1:], strict=False)
        )  # (dim_in, dim_out) for each resolution

        _ResnetBlock = partial(
            ResnetBlock, num_group=num_resnet_block_group, dropout=self.dropout
        )

        # time embedding
        #       time embedding is basically a vector f(t).
        #       it will be used by ResnetBlocks to compute scale and shift.
        #       so the dimension does not have to be the same as model dimension.
        dim_time_emb = dim_base * 4  # default

        if self.type_pos_emb == "learned":
            pos_emb = LearnedSinusoidalPosEmb(dim_learned_pos_emb)
            # dim_fourier: dimension of fourier embedding.
            dim_fourier = dim_learned_pos_emb  # customized dimension
        elif self.type_pos_emb == "random":
            pos_emb = RandomSinusoidalPosEmb(dim_learned_pos_emb)
            dim_fourier = dim_learned_pos_emb
        else:
            pos_emb = SinusoidalPosEmb(dim_base)
            dim_fourier = dim_base  # same as model dimension

        """ computes a vector based on time t,
        used by ResnetBlocks to compute scale and shift
            time_mlp: convert a time t to a vector of dimension dim_time_emb.
                - input: time t, shape: (batch, )
                - output: vector f(t), shape: (batch, dim_time_emb)
                - steps:
                    - pos_emb, shape: (batch, dim_fourier)
                    - mlp, shape: (batch, dim_time_emb)
        """
        self.time_mlp = nn.Sequential(
            pos_emb,
            nn.Linear(dim_fourier, dim_time_emb),
            nn.GELU(),
            nn.Linear(dim_time_emb, dim_time_emb),
        )
        if self.conditioning:
            self.cond_embedder = cond_embedder
            self.post_cond_embedder = nn.Sequential(
                nn.SiLU(),
                nn.Conv1d(dim_base, dim_time_emb, 1),
                Rearrange(
                    "b c l -> b (c l)"
                ),  # (batch, dim_time_emb, 1) -> (batch, dim_time_emb)
            )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_layer = len(list_dim_in_out)  # one layer = multiple sub-layer

        for idx, (d_in, d_out) in enumerate(list_dim_in_out):
            is_last = idx == (num_layer - 1)

            # downsample
            self.downs.append(
                nn.ModuleList(
                    [
                        _ResnetBlock(
                            d_in, d_in, dim_time_emb=dim_time_emb
                        ),  # it has residual connection
                        _ResnetBlock(d_in, d_in, dim_time_emb=dim_time_emb),
                        Residual(
                            PreNorm(
                                d_in,
                                UnetLinearSelfAttention(
                                    d_in,
                                    num_head=num_attn_head,
                                    dim_head=d_in // num_attn_head,
                                ),
                            )
                        ),
                        # if not last, downsample, else use 3x3 conv (same dimension)
                        (
                            Downsample(d_in, d_out)
                            if not is_last
                            else nn.Conv1d(d_in, d_out, 3, padding=1)
                        ),
                    ]
                )
            )  # (batch, d_in, sequence_in) -> (batch, d_out, sequence_out)

        mid_dim = list_dim[-1]  # = last d_out
        self.mid_block1 = _ResnetBlock(mid_dim, mid_dim, dim_time_emb=dim_time_emb)
        self.mid_attn = Residual(
            PreNorm(
                mid_dim,
                UnetAttention(
                    mid_dim, num_head=num_attn_head, dim_head=mid_dim // num_attn_head
                ),
            )
        )
        self.mid_block2 = _ResnetBlock(mid_dim, mid_dim, dim_time_emb=dim_time_emb)

        # upsample
        for idx, (d_in, d_out) in enumerate(reversed(list_dim_in_out)):
            is_last = idx == (num_layer - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        _ResnetBlock(d_in + d_out, d_out, dim_time_emb=dim_time_emb),
                        _ResnetBlock(d_in + d_out, d_out, dim_time_emb=dim_time_emb),
                        Residual(
                            PreNorm(
                                d_out,
                                UnetLinearSelfAttention(
                                    d_out,
                                    num_head=num_attn_head,
                                    dim_head=d_out // num_attn_head,
                                ),
                            )
                        ),
                        (
                            Upsample(d_out, d_in)
                            if not is_last
                            else nn.Conv1d(d_out, d_in, 3, padding=1)
                        ),
                    ]
                )
            )

        # conditioning
        #   pre-final adaln modulation
        #   !not necessary. modulation will be done in the resnet block.

        self.final_resnet_block = _ResnetBlock(
            dim_base * 2, dim_base, dim_time_emb=dim_time_emb
        )  # connect to the input of init_conv
        self.final_conv = nn.Conv1d(dim_base, self.dim_out, 1)  # 1x1 conv

    def forward(
        self,
        x: Float[Tensor, ""],
        time: Float[Tensor, "batch"],
        c: Float[Tensor, "batch num_cond_channel 1"] | None = None,
        force_drop_ids: Float[Tensor, "batch"] | None = None,
        x_self_cond: Float[Tensor, ""] | None = None,
    ) -> Float[Tensor, ""]:
        # generate embedding for time and condition
        encoded_t = self.time_mlp(time)  # shape: (batch, dim_time_emb)
        if self.conditioning and c is not None:
            encoded_c = self.cond_embedder(c, force_drop_ids=force_drop_ids)
            encoded_c = self.post_cond_embedder(
                encoded_c
            )  # shape: (batch, dim_time_emb)
        elif self.conditioning and c is None:
            "if c is not given, then seen as force drop all conditions"
            c = torch.randn(
                (x.shape[0], self.num_cond_channel, 1), device=x.device, dtype=x.dtype
            )
            # shape: (batch, num_cond_channel, 1)
            force_drop_ids = torch.ones(
                (x.shape[0],), device=x.device, dtype=torch.long
            )  # shape: (batch, )
            encoded_c = self.cond_embedder(
                c, force_drop_ids=force_drop_ids
            )  # shape: (batch, dim_base, sequence)
        else:
            encoded_c = torch.zeros_like(encoded_t)  # shape: (batch, dim_time_emb)
        encoded_tc = encoded_t + encoded_c  # shape: (batch, dim_time_emb)

        if self.self_condition:
            if x_self_cond is None:
                warnings.warn(
                    "self_condition is True, \
                        but x_self_cond is None, using full zeros.",
                    stacklevel=2,
                )
                x_self_cond = torch.zeros_like(x)
            x = torch.cat(
                (x, x_self_cond), dim=1
            )  # shape: (batch, num_in_channel * 2, sequence)

        # convert from input channels into base channels (dim_base)
        x = self.init_conv(
            x
        )  # shape: (batch, num_in_channel, sequence) -> (batch, dim_base, sequence)
        x_copy = x.clone()

        h = []  # list of feature maps

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, encoded_tc)  # shape: (batch, dim_in, sequence)
            h.append(x)

            x = block2(x, encoded_tc)  # shape: (batch, dim_in, sequence)
            x = attn(x)
            h.append(x)

            # last downsample does not "downsample"
            x = downsample(x)  # shape: (batch, dim_out, sequence//2)

        x = self.mid_block1(x, encoded_tc)
        x = self.mid_attn(x)
        x = self.mid_block2(x, encoded_tc)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat(
                (x, h.pop()), dim=1
            )  # shape: (batch, dim_in + dim_out, sequence//2)
            x = block1(x, encoded_tc)  # shape: (batch, dim_out, sequence//2)

            x = torch.cat(
                (x, h.pop()), dim=1
            )  # shape: (batch, dim_in + dim_out, sequence//2)
            x = block2(x, encoded_tc)  # shape: (batch, dim_out, sequence//2)
            x = attn(x)  # shape: (batch, dim_out, sequence//2)

            # last upsample does not "upsample"
            x = upsample(x)  # shape: (batch, dim_in, sequence)

        # TODO: this concatenation is unexpected.
        x = torch.cat((x, x_copy), dim=1)
        x = self.final_resnet_block(x, encoded_tc)
        #   shape: (batch, dim_base + dim_base, sequence)
        #           -> (batch, dim_base, sequence)
        x = self.final_conv(x)  # shape: (batch, dim_out, sequence)

        return x

    def forward_with_cfg(
        self,
        x: Float[Tensor, "batch num_in_channel sequence"],
        time: Float[Tensor, "batch"],
        c: Float[Tensor, "batch num_cond_channel 1"] | None = None,
        x_self_cond: Float[Tensor, ""] | None = None,
        cfg_scale: float = 1.0,
    ) -> Float[Tensor, ""]:
        """forward with classfier-free guidance (cfg)

        Inputs:
            - x: (batch, num_in_channel, sequence)
            - time: (batch, )
            - c: None | (batch, num_cond_channel, 1)
            - x_self_cond: None | (batch, num_in_channel, sequence)
            - cfg_scale: float, range [1., +inf), default 1. = no guidance.
                            this scale = 1 + w, the w is as in the paper.
        """
        if c is None or cfg_scale == 1.0:
            return self.forward(x, time, c, x_self_cond)

        _x = torch.cat([x, x], dim=0)  # shape: (batch*2, num_in_channel, sequence)
        _time = torch.cat([time, time], dim=0)  # shape: (batch*2, )
        _c = torch.cat([c, c], dim=0)  # shape: (batch*2, num_cond_channel, 1)
        if x_self_cond is not None:
            _x_self_cond = torch.cat([x_self_cond, x_self_cond], dim=0)
        else:
            _x_self_cond = None
        force_drop_ids_1 = torch.zeros(
            (x.shape[0],), device=x.device, dtype=torch.long
        )  # no drop
        force_drop_ids_2 = torch.ones(
            (x.shape[0],), device=x.device, dtype=torch.long
        )  # drop all
        force_drop_ids = torch.cat(
            [force_drop_ids_1, force_drop_ids_2], dim=0
        )  # shape: (batch*2, )
        # shape: (batch*2, num_cond_channel, sequence), cond + uncond
        cond_x, uncond_x = self.forward(
            _x, _time, _c, force_drop_ids=force_drop_ids, x_self_cond=_x_self_cond
        ).chunk(2, dim=0)
        scaled_x = uncond_x + cfg_scale * (cond_x - uncond_x)

        return scaled_x
