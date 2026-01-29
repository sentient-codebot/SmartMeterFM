import math
import warnings
from collections.abc import Sequence
from functools import partial
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce
from jaxtyping import Complex, Float, Int
from torch import Tensor, nn


# helper functions
def default(value: Any | None, default_value: Any) -> Any:
    if value is not None:
        return value
    else:
        # more efficient in that it does not evaluate default_value unless it is needed
        if callable(default_value):
            return default_value()
        else:
            return default_value


class QuantizeCondition(nn.Module):
    """Quantizes continuous values into discrete levels and provides inverse\
        transformation.
    This module quantizes continuous input values into discrete levels between\
        a specified minimum and maximum value. It can also reconstruct\
        approximate continuous values from the quantized levels through the\
        inverse method.

    Args:
        num_level (int, optional): Number of discrete levels for quantization. Defaults to 400.
        max_val (float, optional): Maximum value of the input range. Defaults to 20000.0.
        min_val (float, optional): Minimum value of the input range. Defaults to 0.0.
    Methods:
        forward(x): Quantizes continuous input values into discrete levels
        inverse(x): Reconstructs continuous values from quantized levels
    Forward Inputs:
        x (torch.Tensor): Continuous input values to be quantized
    Forward Returns:
        torch.Tensor: Discrete levels of the input values

    Example:
        >>> quantizer = QuantizeCondition(num_level=10, max_val=1.0, min_val=0.0)
        >>> x = torch.tensor([0.15, 0.45, 0.75])
        >>> quantized = quantizer(x)  # Forward pass
        >>> reconstructed = quantizer.inverse(quantized)  # Inverse transformation
    """

    def __init__(
        self, num_level: int = 400, max_val: float = 20000.0, min_val: float = 0.0
    ):
        super().__init__()
        self.num_level = num_level
        self.min_val = min_val
        self.max_val = max_val

        if (max_val - min_val) < 1e-6:
            raise ValueError(
                f"max_val {max_val} must be (at least 1e-6) larger \
                    than min_val {min_val}"
            )

        self.level_width = (max_val - min_val) / self.num_level

    def forward(self, x: None | Float[Tensor, "batch "]) -> Int[Tensor, "batch"] | None:
        if x is None:
            return x
        x = torch.clamp(x, self.min_val, self.max_val)
        x = torch.floor((x - self.min_val) / self.level_width).long()
        return x

    def inverse(self, x: Float[Tensor, "batch "]):
        x = x * self.level_width + self.min_val
        return x


class IntegerEmbedder(nn.Module):
    """
    Embed integer suchs as day index or season index into vector representations.
    Also handles label dropout (for classifier-free) guidance?
        (See https://github.com/facebookresearch/DiT/blob/main/models.py#L27)

        for unconditional generation:
            - use dropout > 0. to enable condition dropout (= unconditional generation)
            - set `train=True|None` or use `force_drop_ids` to force drop certain ids

    Arguments:
        - num_embedding: number of embeddings (e.g. 365 for day index)
        - dim_embedding: dimension of embedding
        - dropout: dropout rate for discarding condition (for classifier-free guidance)
        (default: 0.1)
        - quantize: whether to quantize the input (default: False)
        - quantize_max_val: maximum value for quantization (default: 20000.0)
        - quantize_min_val: minimum value for quantization (default: 0.0)

    Input:
        - cond: (batch, )
        - train: bool, whether to use dropout (default: None, use self.training)
        - force_drop_ids: (batch, ), set to 1. to force drop certain ids (default: None)
    Returns:
        - embeddings: (batch, dim_embedding)
    """

    def __init__(
        self,
        num_embedding: int,
        dim_embedding: int,
        dropout: float = 0.1,
        quantize: bool = False,
        quantize_max_val: float = 20000.0,
        quantize_min_val: float = 0.0,
    ):
        super().__init__()
        use_null_embedding = dropout > 0  # drop condition = unconditional generation
        self.num_embedding = num_embedding
        self.dim_embedding = dim_embedding
        self.dropout = dropout
        self.quantize = quantize
        self.quantize_level = num_embedding
        self.quantize_max_val = quantize_max_val
        self.quantize_min_val = quantize_min_val

        if self.quantize:
            self.quantizer = QuantizeCondition(
                num_level=self.quantize_level,
                max_val=self.quantize_max_val,
                min_val=self.quantize_min_val,
            )
        self.embedding_table = nn.Embedding(
            num_embedding + use_null_embedding, dim_embedding
        )

    def token_drop(
        self,
        cond: Tensor,
        force_drop_ids: Tensor | None = None,
    ):
        """
        Drop condition to enable classifier-free guidance.

        `cond`: condition, shape: (batch, ), range: [0, num_embedding)
        `force_drop_ids`: force drop certain ids, shape: (batch, ), range: {0, 1}

        """
        if force_drop_ids is None:
            if self.training:
                drop_ids = torch.rand(cond.shape[0], device=cond.device) < self.dropout
            else:
                drop_ids = torch.zeros(
                    cond.shape[0], device=cond.device, dtype=torch.bool
                )
        else:
            drop_ids = force_drop_ids == 1

        # if drop all, return all null embedding, `cond` is not used
        null_idx = torch.full(
            (drop_ids.shape[0],),
            self.num_embedding,
            device=drop_ids.device,
            dtype=torch.int64,
        )
        # shape (batch, )
        result = torch.where(
            drop_ids,
            null_idx,
            cond,
        )

        return result

    def forward(
        self,
        cond: Float[Tensor, "batch"],
        force_drop_ids: Float[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch dim_embedding"]:
        if self.quantize:
            cond = self.quantizer(cond)

        use_dropout = self.dropout > 0.0
        if use_dropout:
            cond = self.token_drop(cond, force_drop_ids)

        embeddings = self.embedding_table(cond.long())  # shape: (batch, dim_embedding)
        return embeddings

    def get_null_token_value(self) -> int:
        """Return the null token value for this embedder (used when condition should be dropped)."""
        return self.num_embedding


class Zeros(nn.Module):
    """A PyTorch module that returns a tensor of zeros with specified dimensions.
    This module creates a tensor of zeros with shape (batch_size, dim_embedding) based
    on the input condition tensor's batch size. The output tensor will have the same
    device and dtype as the input condition tensor.
    Args:
        num_embedding (int): Number of embeddings (not used in forward pass but kept for
            consistency with other embedding modules).
        dim_embedding (int): The dimension of the output embedding vectors.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Inputs:
        cond (torch.Tensor): A tensor of shape (batch_size, *) representing the input
            condition.
    Returns:
        torch.Tensor: A tensor of zeros with shape (batch_size, dim_embedding).
    """

    def __init__(self, num_embedding: int, dim_embedding: int, *args, **kwargs):
        super().__init__()
        self.num_embedding = num_embedding
        self.dim_embedding = dim_embedding

    def forward(self, cond, *args, **kwargs):
        batch_size = cond.shape[0]
        return torch.zeros(
            batch_size, self.dim_embedding, device=cond.device, dtype=cond.dtype
        )


# deprecated
class ConcatEmbedder(nn.Module):
    """Automatically splits input into multiple parts and SUMS each part together.
        Sum up the embeddings and returns.

    Arguments:
        - embedder_list: list of embedders
            - either (batch, ) -> (batch, dim_embedding)

    Input:
        - label: (batch, num_labels), num_labels == len(embedder_list)

    Return:
        - encoded_label: (batch, dim_base)
    """

    def __init__(self, list_embedder: Sequence[IntegerEmbedder]):
        super().__init__()
        self.list_embedder = nn.ModuleList(list_embedder)
        # dim_embeddings must be equal
        if len({embedder.dim_embedding for embedder in self.list_embedder}) != 1:
            raise ValueError("dim_embedding must be equal for all embedders")
        self.dim_embedding = self.list_embedder[0].dim_embedding

    def forward(
        self, label: Float[Tensor, "batch num_labels"], **kwargs
    ) -> Float[Tensor, "batch dim_base"]:
        if label.shape[1] != len(self.list_embedder):
            raise ValueError("label shape[1] must be equal to len(embedder_list)")

        list_label = label.split(1, dim=1)  # list of (batch, 1)
        sum_encoded_label = 0.0
        for label, embedder in zip(list_label, self.list_embedder, strict=False):
            encoded_label = embedder(
                label.squeeze(-1), **kwargs
            )  # shape: (batch, dim_embedding)
            sum_encoded_label += encoded_label
        return sum_encoded_label


class CombinedEmbedder(nn.Module):
    """Automatically parse a dictionary of conditions, get their embeddings\
        and sum the embeddings together.

    Arguments:
        - embedder_dict dict[str, Callable]: dictionary of embedders, each callable:
            - (batch, ) -> (batch, dim_embedding)

    Input:
        - label: dict[str, Tensor], each tensor (batch, num_labels), \
            num_labels == len(embedder_list)

    Return:
        - encoded_label: (batch, dim_base)
    """

    def __init__(self, dict_embedder: dict[str, IntegerEmbedder]):
        super().__init__()
        self.dict_embedder = nn.ModuleDict(dict_embedder)
        self.embedder_keys = list(self.dict_embedder.keys())
        # dim_embeddings must be equal
        set_dim_embedding = {
            embedder.dim_embedding
            if hasattr(embedder, "dim_embedding")
            else embedder.dim_base
            for embedder in self.dict_embedder.values()
        }
        self.dim_embedding = next(iter(set_dim_embedding))  # get first value
        if len(set_dim_embedding) != 1:
            raise ValueError("dim_embedding must be equal for all embedders")

    def forward(
        self,
        dict_labels: dict[str, Float[Tensor, "batch"]] | None,
        dict_extra: dict[str, Any] | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> Float[Tensor, "batch dim_base"]:
        if dict_labels is None or len(dict_labels) == 0:
            # Create null token tensors for each embedder
            device = default(device, next(self.parameters()).device)
            if batch_size is None:
                batch_size = 1
            dict_labels = {}
            for label_name in self.embedder_keys:
                embedder = self.dict_embedder[label_name]
                # For IntegerEmbedder and similar, null token is num_embedding
                null_token_idx = getattr(embedder, "num_embedding", 0)
                dict_labels[label_name] = torch.full(
                    (batch_size,), null_token_idx, device=device, dtype=torch.long
                )
        else:
            device = getattr(next(iter(dict_labels.values())), "device", device)
            if device is None:
                device = next(self.parameters()).device

        if dict_extra is None:
            dict_extra = {}

        sum_encoded_label = torch.zeros(
            batch_size,
            self.dim_embedding,
            device=device,
        )
        for label_name in self.embedder_keys:
            label_value = dict_labels[label_name]  # Always expect tensor, no None check
            label_value = label_value.squeeze(-1)
            extras = dict_extra.get(label_name, {})
            emb_label = self.dict_embedder[label_name](
                label_value, **extras
            )  # shape (batch, dim_embedding)
            sum_encoded_label += emb_label

        return sum_encoded_label


class ContextEmbedder(nn.Module):
    """Automatically parse a dictionary of conditions, get their embeddings\
        and concate them into a sequence of embeddings.

    Arguments:
        - embedder_dict dict[str, Callable]: dictionary of embedders, each callable:
            - (batch, ...) -> (batch, #seq, dim_embedding)

    Input:
        - label: dict[str, Tensor], each tensor (batch, num_labels), \
            num_labels == len(embedder_list)

    Return:
        - label_sequence: (batch, #sum_seq, dim_base)
    """

    def __init__(self, dict_embedder: dict[str, nn.Module]):
        super().__init__()
        self.dict_embedder = nn.ModuleDict(dict_embedder)
        self.embedder_keys = list(self.dict_embedder.keys())
        # dim_embeddings must be equal
        set_dim_embedding = {
            embedder.dim_embedding
            if hasattr(embedder, "dim_embedding")
            else embedder.dim_base
            for embedder in self.dict_embedder.values()
        }
        if len(set_dim_embedding) != 1:
            raise ValueError("dim_embedding must be equal for all embedders")

    def _parse_input(
        self,
        dict_labels,
        dict_extra: dict | None,
        batch_size: int = 1,
        device: torch.device = None,
    ) -> tuple[dict, dict]:
        if dict_labels is None:
            # Create null token tensors for each embedder
            device = device or next(self.parameters()).device
            dict_labels = {}
            for label_name in self.dict_embedder.keys():
                embedder = self.dict_embedder[label_name]
                # For IntegerEmbedder and similar, null token is num_embedding
                null_token_idx = getattr(embedder, "num_embedding", 0)
                dict_labels[label_name] = torch.full(
                    (batch_size,), null_token_idx, device=device, dtype=torch.long
                )

        if dict_extra is None:
            dict_extra = {}

        return dict_labels, dict_extra

    def _to_3d(self, tensor):
        if tensor.ndim == 2:
            return tensor.unsqueeze(1)
        elif tensor.ndim == 3:
            return tensor
        else:
            raise ValueError(f"tensor must be 2D or 3D, but got {tensor.ndim}D tensor")

    def forward(
        self,
        dict_labels: dict[str, Float[Tensor, "batch *"]] | None,
        dict_extra: dict[str, Any] | None = None,
    ) -> Float[Tensor, "batch seq dim_base"]:
        """if force drop ids, null embedding or just dropout in the final seq?"""
        # Determine batch_size and device from input or use defaults
        if dict_labels is not None and len(dict_labels) > 0:
            first_tensor = next(iter(dict_labels.values()))
            batch_size = first_tensor.shape[0] if first_tensor is not None else 1
            device = (
                first_tensor.device
                if first_tensor is not None
                else next(self.parameters()).device
            )
        else:
            batch_size = 1
            device = next(self.parameters()).device

        dict_labels, dict_extra = self._parse_input(
            dict_labels, dict_extra, batch_size, device
        )
        list_embs = []

        for label_name in self.embedder_keys:
            label_value = dict_labels[label_name]  # Always expect tensor, no None check
            label_value = label_value.squeeze(-1)
            extras = dict_extra.get(label_name, {})
            emb_label = self.dict_embedder[label_name](
                label_value, **extras
            )  # shape (batch, dim_embedding)
            emb_label = self._to_3d(emb_label)  # ensure 3D tensor
            list_embs.append(emb_label)

        label_sequence = torch.cat(
            list_embs, dim=1
        )  # shape (batch, #seq, dim_embedding)
        return label_sequence


class EmbedderWrapper(nn.Module):
    """
    Automatically splits input into multiple parts and SUMS each part together.
        Concatenates the embeddings and returns.

    Arguments:
        - embedder_list: list of embedders
            - either (batch, ) -> (batch, dim_embedding)
            - or (batch, channel) -> (batch, dim_embedding)
        - list_dim: list of dimensions for each condition (num_condiiton_i_channel)

    Input:
        - c: (batch, *, num_condiiton_channel)
        - *args: additional arguments for embedders, such as `train` or `force_drop_ids`

    Return:
        - encoded_c: (batch, *, dim_base)

    """

    def __init__(
        self,
        list_embedder: Sequence[IntegerEmbedder],
        list_dim_cond: Sequence[int],
    ):
        super().__init__()
        assert len(list_embedder) == len(
            list_dim_cond
        ), "embedder_list and dim_list must have same length"
        self.list_embedder = (
            nn.ModuleList(list_embedder)
            if not isinstance(list_embedder, nn.ModuleList)
            else list_embedder
        )
        self.list_dim_cond = list_dim_cond

    def forward(
        self, c: Float[Tensor, "batch num_condition_channel #foo"], *args, **kwargs
    ) -> Float[Tensor, "batch dim_base 1"]:
        list_c = c.split(self.list_dim_cond, dim=1)  # list of (batch, dim, 1)

        # list_encoded_c = []
        sum_encoded_c = 0.0
        for cond, embedder in zip(list_c, self.list_embedder, strict=False):
            if cond.shape[1] == 1:  # channel == 1, shape (batch, 1, 1)
                cond = rearrange(cond, "batch 1 1 -> batch")
                encoded_c = embedder(
                    cond, *args, **kwargs
                )  # shape: (batch, dim_embedding)
            else:  # channel > 1, shape (batch, channel, 1)
                cond = rearrange(cond, "batch channel 1 -> batch channel")
                encoded_c = embedder(
                    cond, *args, **kwargs
                )  # shape: (batch, channel) -> (batch, dim_embedding)
            # list_encoded_c.append(encoded_c)
            sum_encoded_c += encoded_c  # shape: (batch, dim_embedding)

        # concat_encoded_c = torch.cat(list_encoded_c, dim=1) # shape: (batch, dim_all)

        return rearrange(sum_encoded_c, "batch channel -> batch channel 1")


# small modules
class SinusoidalPosEmb(nn.Module):
    """for position t, dimension i of a d-dim vector, the embedding is
    1/(10000**(i/(d/2-1)))

    dim must be even.

    Args:
        - dim (int): dimension of the embedding, must be even
        - scaling_factor (float): `pos <- pos * scaling_factor`. default = 1.
    """

    def __init__(self, dim: int, scaling_factor: float = 1.0):
        super().__init__()
        assert dim % 2 == 0, "dimension must be even"
        self.dim = dim
        self.scaling_factor = scaling_factor

    def forward(self, pos: Float[Tensor, "batch"]) -> Float[Tensor, "batch dim"]:
        pos = pos * self.scaling_factor
        device = pos.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # shape: (dim/2,)
        emb = pos.unsqueeze(-1) * emb.unsqueeze(
            0
        )  # shape: (batch, 1) * (1, dim/2) -> (batch, dim/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch, dim)
        return emb


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Complex[Tensor, "seq_len dim//2"]:
    """precompute the rotatory complex numbers for RoPE. origin: Llama

    Args:
        dim (int): dimension of the embedding, must be even
        max_seq_len (int): maximum sequence length
        theta (float): scaling factor for the frequency. default = 10000.0
    Returns:
        freqs_cis (Complex[Tensor, "seq_len dim//2"]): precomputed complex numbers

    """
    # shape: (dim//2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # shape: (max_seq_len)
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: Float[Tensor, "batch * seq_len dim"],
    freqs_cis: Complex[Tensor, "seq_len dim//2"] | Complex[Tensor, "batch seq_len dim"],
    xk: torch.Tensor | None = None,
):
    """Apply rotary embedding to the input tensor.

    Args:
        xq (Float[Tensor, "batch * dim"]): input tensor
        freqs_cis (Complex[Tensor, "seq_len dim//2"] |
            Complex[Tensor, "batch seq_len dim//2]): precomputed complex numbers.
            assumes on the same device as xq. Can be batched.
        xk (torch.Tensor | None): optional second input tensor to apply rotary embedding
    Returns: tuple[Tensor, Tensor]
        xq (Float[Tensor, "batch seq_len dim"]): output tensor with rotary embedding applied
        xk (torch.Tensor | None): optional second output tensor with rotary embedding applied

    """
    # splits the last dimension into pairs -> complex
    # shape [batch, *, seq_len, dim//2]
    assert xq.shape[-1] % 2 == 0, "last dimension must be even"
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    if xk is not None:
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    while freqs_cis.ndim < xq_.ndim:
        # batch dimension preserved if exists
        freqs_cis = freqs_cis.unsqueeze(-3)  # [..., seq_len, dim//2]
    # shape [batch, *, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2).type_as(xq)
    if xk is not None:
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2).type_as(xk)
    else:
        xk_out = None
    return xq_out, xk_out


class RotaryPosEmb(nn.Module):
    """Apply rotary pos embedding to the input tensor.

    Args:
        dim (int): dimension of the embedding, must be even
        max_seq_len (int): maximum sequence length, default = 4096

    """

    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        assert dim % 2 == 0, "dimension must be even"
        self.freqs_cis = precompute_freqs_cis(
            dim=dim,
            max_seq_len=max_seq_len,
        )

    def forward(self, seq: Float[Tensor, "batch * seq_len dim"], start_pos: int = 0):
        """Apply rotary embedding to the input tensor.

        Args:
            seq (Float[Tensor, "batch * seq_len dim"]): input tensor
            start_pos (int): starting position for the rotary embedding, default = 0
        Returns:
            seq (Float[Tensor, "batch * seq_len dim"]): output tensor with rotary embedding applied

        """
        self.freqs_cis = self.freqs_cis.to(seq.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq.shape[-2]]
        seq = apply_rotary_emb(seq, freqs_cis)[0]
        return seq


class RandomSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """  # noqa

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dimension must be even"
        half_dim = dim // 2 - 1
        self.weights = nn.Parameter(
            torch.randn(1, half_dim), requires_grad=False
        )  # random

    def forward(self, pos: Float[Tensor, "batch"]) -> Float[Tensor, "batch dim"]:
        pos = rearrange(pos, "b -> b 1")
        freqs = pos * self.weights * 2 * math.pi  # shape: (batch, dim/2-1)
        fouriered = torch.cat(
            (freqs.sin(), freqs.cos()), dim=-1
        )  # shape: (batch, dim-2)
        fouriered = torch.cat((pos, fouriered), dim=-1)  # shape: (batch, dim-1)
        fouriered = torch.cat((-pos, fouriered), dim=-1)  # shape: (batch, dim)
        return fouriered


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dimension must be even"
        half_dim = dim // 2 - 1
        self.weights = nn.Parameter(
            torch.randn(1, half_dim), requires_grad=True
        )  # leranable

    def forward(self, pos: Float[Tensor, "batch"]) -> Float[Tensor, "batch dim"]:
        pos = rearrange(pos, "b -> b 1")
        freqs = pos * self.weights * 2 * math.pi  # shape: (batch, dim/2-1)
        fouriered = torch.cat(
            (freqs.sin(), freqs.cos()), dim=-1
        )  # shape: (batch, dim-2)
        fouriered = torch.cat((pos, fouriered), dim=-1)  # shape: (batch, dim-1)
        fouriered = torch.cat((-pos, fouriered), dim=-1)  # shape: (batch, dim)
        return fouriered


class PositionEmbedder(nn.Module):
    """Position (year/month) Embedder
    Maps (year/month, Tensor[int]) -> (embedding, high-dimensional vector).
    Uses sinusoidal embedding + mlp layer.

    Args:
        dim_base (int): the target dimension for embedding.

    Input:
        pos (tensor, int): [batch]

    Output:
        embedding: [batch dim_base]

    """

    def __init__(self, dim_embedding: int, dropout: float = 0.1):
        super().__init__()
        use_null_embedding = dropout > 0
        self.dropout = dropout
        self.dim_embedding = dim_embedding
        pos_emb = SinusoidalPosEmb(self.dim_embedding)
        mlp = nn.Sequential(
            nn.Linear(self.dim_embedding, self.dim_embedding),
            nn.SiLU(),
            nn.Linear(self.dim_embedding, self.dim_embedding),
        )
        self.layers = nn.Sequential(
            pos_emb,  # shape [batch dim_base]
            mlp,  # shape [batch dim_base]
        )
        if use_null_embedding:
            self.null_embedding = nn.Embedding(1, dim_embedding)

    def token_drop(self, true_emb: Float[Tensor, "batch dim_emb"], force_drop_ids=None):
        """
        Drop condition to enable classifier-free guidance.

        `true_emb`: condition, shape: (batch, dim_emb)
        `force_drop_ids`: force drop certain ids, shape: (batch, ), range: {0, 1}

        """
        if not hasattr(self, "null_embedding"):
            return true_emb
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(true_emb.shape[0], device=true_emb.device) < self.dropout
            )
        else:
            drop_ids = force_drop_ids == 1
        null_emb = self.null_embedding(
            torch.zeros(true_emb.shape[0], device=true_emb.device, dtype=torch.int64)
        )
        # [batch] -> [batch, dim_emb]
        mixed_emb = torch.where(drop_ids.unsqueeze(1), null_emb, true_emb)
        # dropped cond are set to num_embedding (last category)
        return mixed_emb

    def forward(
        self,
        pos: Int[Tensor, "batch"] | None,
        train: bool | None = None,
        force_drop_ids: Float[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch dim_base"]:
        if pos is None:
            if force_drop_ids is None or force_drop_ids.sum() < force_drop_ids.shape[0]:
                raise ValueError(
                    "pos is None, force_drop_ids must be provided and must be all ones."
                )
            return self.null_embedding(
                torch.zeros(
                    force_drop_ids.shape[0],
                    device=force_drop_ids.device,
                    dtype=torch.int64,
                )
            )
        true_emb = self.layers(pos)
        train = train if train is not None else self.training
        use_dropout = self.dropout > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            mixed_emb = self.token_drop(true_emb, force_drop_ids)
            return mixed_emb
        return true_emb

    def get_null_token_value(self) -> int:
        """Return the null token value for this embedder (dummy value, relies on force_drop_ids)."""
        return 0  # PositionEmbedder uses dummy value + force_drop_ids to handle nulls


# building blocks
class AttentionOutput(NamedTuple):
    """namedtuple
    - x_out: Float[Tensor, "batch sequence dim"]
    - context_out: Float[Tensor, "batch seq2 dim"] | None = None
    """

    x_out: Float[Tensor, "batch sequence dim"]
    context_out: Float[Tensor, "batch seq2 dim"] | None = None


def linear_attn_op(
    query: Float[Tensor, "batch head seq_q dim"],
    key: Float[Tensor, "batch head seq_k dim"],
    value: Float[Tensor, "batch head seq_k dim"],
    attn_mask: Float[Tensor, "batch head seq_q seq_k"] | None = None,
):
    """Linear attention operation. replacement for
    `scaled_dot_product_attention` with linear complexity.

    Args:

    Returns:
    """
    # hparams
    kernel_func = nn.ReLU(inplace=False)
    eps = 1e-6
    pad_val = 1.0
    query = kernel_func(query)  # shape: (batch, head, seq_q, dim)
    key = kernel_func(key)  # shape: (batch, head, seq_k, dim)

    if attn_mask is not None:
        # q_mask = attn_mask[:, :, :, 0]  # shape: (B, H, L_q)
        k_mask = attn_mask[:, :, 0, :]  # shape: (B, H, L_k)
        # q_mask = q_mask.unsqueeze(-1)  # shape: (B, H, L_q, 1)
        k_mask = k_mask.unsqueeze(-1)  # shape: (B, H, L_k, 1)
        # query = query * q_mask  # shape: (B, H, L_q, D)
        key = key * k_mask  # shape: (B, H, L_k, D)

    value = F.pad(
        value,
        (0, 1),
        mode="constant",
        value=pad_val,
    )  # shape: (B, H, L_k, D) -> (B, H, L_k, D+1)

    vk = einsum(
        value,
        key,
        "b h L_k d_plus_one, b h L_k d -> b h d_plus_one d",
    )  # shape: (batch, head, D_v+1, D_k)
    result = einsum(
        vk,
        query,
        "b h d_plus_one d, b h L_q d -> b h L_q d_plus_one",
    )  # shape: (batch, head, seq_q, D_v+1)
    # normalization
    result = result[:, :, :, :-1] / (result[:, :, :, -1:] + eps)  # (B, H, seq_Q, D_v)

    return result  # shape: (batch, head, seq_q, dim)


class MixFFN(nn.Module):
    """MixFFN as a replacement for the MLP in the transformer block.

    Args:

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.in_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim * 2,
            kernel_size=1,
            bias=True,
        )
        self.depth_conv = nn.Conv1d(
            in_channels=dim * 2,
            out_channels=dim * 2,
            kernel_size=3,
            padding=1,
            bias=True,
            groups=dim * 2,  # depthwise convolution
        )
        self.out_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            bias=True,
        )
        self.activation = nn.SiLU()  # or nn.GELU()

    def forward(
        self,
        x: Float[Tensor, "batch sequence dim"],
    ) -> Float[Tensor, "batch sequence dim"]:
        x = x.transpose(1, 2)  # shape: (batch, dim, sequence)
        x = self.in_conv(x)  # shape: (batch, dim*2, sequence)
        x = self.depth_conv(x)  # shape: (batch, dim*2, sequence)

        x, gate = x.chunk(
            2, dim=1
        )  # shape: (batch, dim, sequence), (batch, dim, sequence)
        gate = self.activation(gate)  # shape: (batch, dim, sequence)
        x = x * gate  # shape: (batch, dim, sequence)

        x = self.out_conv(x)  # shape: (batch, dim, sequence)
        x = x.transpose(1, 2)  # shape: (batch, sequence, dim)

        return x  # shape: (batch, sequence, dim)


class RMSNormSelfAttention(nn.Module):
    """Self Attention with RMS Norm

    Arguments:
        dim (int): model dimension (==input output dimension)
        num_head (int): number of heads, `dim` must be divisible by `num_head`

    Input:
        x (Float[Tensor, "batch sequence dim"]): input tensor

    Output:
        x (Float[Tensor, "batch sequence dim"]): output tensor

    """

    def __init__(
        self, dim: int, num_head: int = 4, max_seq_len: int = 4096, linear: bool = False
    ):
        super().__init__()
        self.dim_model = dim
        self.num_head = num_head
        if dim % num_head != 0:
            raise ValueError("dim must be divisible by num_head")
        self.dim_head = dim // num_head
        self.scale = self.dim_head**-0.5
        self.to_qkv = nn.Linear(self.dim_model, self.dim_model * 3, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(self.dim_model, self.dim_model),
            nn.RMSNorm(self.dim_model),
        )
        self.rms_norm_q = nn.RMSNorm(self.dim_model)
        self.rms_norm_k = nn.RMSNorm(self.dim_model)
        if self.dim_head % 2 != 0:
            raise ValueError("dim_head must be even for rotary embedding")
        self.freqs_cis = precompute_freqs_cis(
            dim=self.dim_head,
            max_seq_len=max_seq_len,
        )
        self.linear = linear

    def forward(
        self,
        x: Float[Tensor, "batch sequence dim"],
        attn_mask: Float[Tensor, "batch sequence sequence"] | None = None,
        start_pos: Int[Tensor, "batch"] | None = None,
    ) -> AttentionOutput:
        if start_pos is None:
            start_pos = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3 * (batch, sequence, dim)
        q, k, v = (t for t in qkv)  # shape: 3 * (batch, sequence, dim)
        q = self.rms_norm_q(q)  # shape: (batch, sequence, dim)
        k = self.rms_norm_k(k)  # shape: (batch, sequence, dim)
        q, k, v = (
            rearrange(
                t,
                "b L (num_head dim_head) -> b num_head L dim_head",
                num_head=self.num_head,
            )
            for t in (q, k, v)
        )  # shape: 3 * (batch, num_head, sequence, dim_head)
        self.freqs_cis = self.freqs_cis.to(q.device)
        # quickly calculate indices for freqs_cis
        _offset = torch.arange(
            q.shape[-2], device=start_pos.device, dtype=start_pos.dtype
        ).unsqueeze(0)  # (1, seq_len)
        pos_indices = start_pos.unsqueeze(1) + _offset  # (batch, seq_len)
        freqs_cis = self.freqs_cis[pos_indices]  # (batch, seq_len, dim//2)
        q, k = apply_rotary_emb(q, freqs_cis, k)

        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, "b L1 L2 -> b 1 L1 L2").bool()

        if not self.linear:
            attn_out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                attn_mask=attn_mask,
            )  # shape: (batch, num_head, sequence, dim_head)
        else:
            attn_out = linear_attn_op(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
            )  # shape: (batch, num_head, seq_q, dim_value_head)
        attn_out = rearrange(
            attn_out, "b num_head L dim_head -> b L (num_head dim_head)"
        )  # shape: (batch, sequence, dim)
        final_out = AttentionOutput(
            self.to_out(attn_out)
        )  # shape: (batch, sequence, dim)
        return final_out


class RMSNormSelfCrossAttention(nn.Module):
    """Attention with RMS Norm and Cross Attention with context.
    NOTE: no positional embedding is applied on the context. assuming they are
    order-agnostic. each context element's representation is learned in previous
    layers.

    Arguments:
        dim (int): model dimension (==input output dimension)
        num_head (int): number of heads, `dim` must be divisible by `num_head`
        use_context (bool): whether to use context

    Input:
        x (Float[Tensor, "batch sequence dim"]): input tensor
        c (Float[Tensor, "batch seq2 dim"] | None): context tensor

    Output:
        x (Float[Tensor, "batch sequence dim"]): output tensor
        c (Float[Tensor, "batch seq2 dim"] | None): output tensor
    """

    def __init__(
        self,
        dim: int,
        num_head: int = 4,
        use_context: bool = False,
        max_seq_len: int = 4096,
        linear: bool = False,
    ):
        super().__init__()
        self.dim_model = dim
        self.num_head = num_head
        if dim % num_head != 0:
            raise ValueError("dim must be divisible by num_head")
        self.dim_head = dim // num_head
        self.scale = self.dim_head**-0.5
        self.to_qkv = nn.Linear(self.dim_model, self.dim_model * 3, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(self.dim_model, self.dim_model),
            nn.RMSNorm(self.dim_model),
        )
        self.rms_norm_q = nn.RMSNorm(self.dim_model)
        self.rms_norm_k = nn.RMSNorm(self.dim_model)

        self.use_context = use_context
        if self.use_context:
            self.c_to_qkv = nn.Linear(self.dim_model, self.dim_model * 3, bias=True)
            self.c_rms_norm_q = nn.RMSNorm(self.dim_model)
            self.c_rms_norm_k = nn.RMSNorm(self.dim_model)
            self.c_to_out = nn.Sequential(
                nn.Linear(self.dim_model, self.dim_model),
                nn.RMSNorm(self.dim_model),
            )
        if self.dim_head % 2 != 0:
            raise ValueError("dim_head must be even for rotary embedding")
        self.freqs_cis = precompute_freqs_cis(
            dim=self.dim_head,
            max_seq_len=max_seq_len,
        )
        self.linear = linear

    def _get_augmented_mask(
        self, attn_mask: Float[Tensor, "batch seq1 seq2"], context_length: int
    ):
        """`attn_mask` without context -> `attn_mask` with context.
        if sequence length == L, context length == C, then:
            (b L L) -> (b L+C L+C)
        """
        attn_mask = attn_mask.detach().bool()  # make sure no grad
        batch_size = attn_mask.shape[0]
        _seq_length = attn_mask.shape[1]  # [b, L, L]
        block_c_to_seq = attn_mask.any(dim=1, keepdim=True)  # [b, 1, L]
        block_c_to_seq = block_c_to_seq.expand(-1, context_length, -1)  # [b, C, L]
        block_seq_to_c = attn_mask.any(dim=2, keepdim=True).expand(
            -1, -1, context_length
        )  # [b, L, C]
        block_c_to_c = torch.ones(
            (batch_size, context_length, context_length),
            device=attn_mask.device,
            dtype=attn_mask.dtype,
        )  # [b, C, C]
        block_seq_to_both = torch.concat(
            [attn_mask, block_seq_to_c], dim=2
        )  # [b, L, L+C]
        block_c_to_both = torch.concat(
            [block_c_to_seq, block_c_to_c], dim=2
        )  # [b, C, L+C]
        full_mask = torch.concat(
            [block_seq_to_both, block_c_to_both], dim=1
        )  # [b, L+C, L+C]

        return full_mask

    def forward(
        self,
        x: Float[Tensor, "batch sequence dim"],
        context: Float[Tensor, "batch seq2 dim"] | None = None,
        attn_mask: Float[Tensor, "batch sequence sequence"] | None = None,
        start_pos: Int[Tensor, "batch"] | None = None,
    ) -> AttentionOutput:
        if start_pos is None:
            start_pos = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3 * (batch, sequence, dim)
        q, k, v = (t for t in qkv)  # shape: 3 * (batch, sequence, dim)
        q = self.rms_norm_q(q)  # shape: (batch, sequence, dim)
        k = self.rms_norm_k(k)  # shape: (batch, sequence, dim)
        q, k, v = (
            rearrange(
                t,
                "b L (num_head dim_head) -> b num_head L dim_head",
                num_head=self.num_head,
            )
            for t in (q, k, v)
        )  # shape: 3 * (batch, num_head, sequence, dim_head)
        self.freqs_cis = self.freqs_cis.to(q.device)
        # quickly calculate indices for freqs_cis
        _offset = torch.arange(
            q.shape[-2], device=start_pos.device, dtype=start_pos.dtype
        ).unsqueeze(0)  # (1, seq_len)
        pos_indices = start_pos.unsqueeze(1) + _offset  # (batch, seq_len)
        freqs_cis_x = self.freqs_cis[pos_indices]  # (batch, seq_len, dim//2)
        q, k = apply_rotary_emb(q, freqs_cis_x, k)
        if self.use_context and context is not None:
            c_qkv = self.c_to_qkv(context).chunk(3, dim=-1)  # 3 * (batch, seq2, dim)
            c_q, c_k, c_v = (t for t in c_qkv)  # shape: 3 * (batch, seq2, dim)
            c_q = self.c_rms_norm_q(c_q)  # shape: (batch, seq2, dim)
            c_k = self.c_rms_norm_k(c_k)
            c_q, c_k, c_v = (
                rearrange(
                    t,
                    "b L (num_head dim_head) -> b num_head L dim_head",
                    num_head=self.num_head,
                )
                for t in (c_q, c_k, c_v)
            )  # shape: 3 * (batch, num_head, seq2, dim_head)
            q = torch.concat((q, c_q), dim=2)
            # shape: (batch, num_head, sequence+seq2, dim_head)
            k = torch.concat((k, c_k), dim=2)
            # shape: (batch, num_head, sequence+seq2, dim_head)
            v = torch.concat((v, c_v), dim=2)
            # shape: (batch, num_head, sequence+seq2, dim_head)

        if attn_mask is not None:
            if context is not None:
                attn_mask = self._get_augmented_mask(
                    attn_mask, context.shape[1]
                )  # [batch, L+C, L+C]
            attn_mask = rearrange(attn_mask, "b L1 L2 -> b 1 L1 L2").bool()

        if not self.linear:
            attn_out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                attn_mask=attn_mask,
            )  # shape: (B, #H, L, dim_head)
        else:
            attn_out = linear_attn_op(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
            )  # shape: (B, #H, L, dim_head)

        attn_out = rearrange(
            attn_out, "b num_head L dim_head -> b L (num_head dim_head)"
        )  # shape: (batch, sequence, dim)
        x_out = attn_out[:, : x.shape[1], :]  # shape: (batch, sequence, dim)
        if self.use_context and context is not None:
            c_out = attn_out[:, x.shape[1] :, :]  # shape: (batch, seq2, dim)
            return AttentionOutput(self.to_out(x_out), self.c_to_out(c_out))
        else:
            return AttentionOutput(self.to_out(x_out))


# backbone modules
class TimeEmbedder(nn.Module):
    """Time Embedder

    Arguments:
        - dim_base: d_model of transformer
        - type_pos_emb: sinusoidal | learned | random

    Forward:
        input:
            - time: (batch, ) range (0, 1)

        return:
            - time_emb: (batch, dim_base)
    """

    def __init__(self, dim_base: int, type_pos_emb: str = "sinusoidal"):
        super().__init__()
        self.dim_base = dim_base
        self.type_pos_emb = type_pos_emb
        if self.type_pos_emb == "learned":
            pos_emb = LearnedSinusoidalPosEmb(self.dim_base)
        elif self.type_pos_emb == "random":
            pos_emb = RandomSinusoidalPosEmb(self.dim_base)
        elif self.type_pos_emb == "sinusoidal":
            pos_emb = SinusoidalPosEmb(self.dim_base, scaling_factor=1000.0)
        else:
            raise ValueError("positional embedding type must be one of learned, random")
        self.time_mlp = nn.Sequential(
            pos_emb,
            nn.Linear(self.dim_base, self.dim_base),
            nn.SiLU(),
            nn.Linear(self.dim_base, self.dim_base),
        )

    def forward(self, time: Float[Tensor, "batch"]) -> Float[Tensor, "batch dim"]:
        return self.time_mlp(time)


class DenoisingTransformer(nn.Module):
    """transformer for 1D data
    Float[Tensor, ""] -> Float[Tensor, ""]

    Arguments:
        - dim_base: d_model of transformer, recommended value = ?
        - ...
        - type_transformer: transformer or gpt2
        - conditioning: whether to use conditioning,
        the condition tensor 'c' should have same dimension as `dim_base`

    Forward:
        input:
            - x: (batch, num_in_channel, sequence)
            - time: (batch, )
            - x_self_cond: None|(batch, num_in_channel, sequence)

        return:
            - x: (batch, dim_out, sequence)

    """

    def __init__(
        self,
        dim_base: int,
        num_in_channel: int = 3,
        dim_out: int | None = None,
        self_condition: bool = False,
        type_pos_emb: str = "sinusoidal",
        # dim_learned_pos_emb: int = 16,
        num_attn_head: int = 4,
        num_encoder_layer=6,
        num_decoder_layer=6,
        dim_feedforward=2048,  # why so big by default?
        dropout=0.1,
        learn_variance: bool = False,
        label_embedder: CombinedEmbedder | None = None,
        context_embedder: ContextEmbedder | None = None,
    ):
        super().__init__()
        self.dim_base = dim_base
        self.num_in_channel = num_in_channel
        self.self_condition = self_condition
        self.type_pos_emb = type_pos_emb
        self.num_atten_head = num_attn_head
        self.num_encoder_layer = num_encoder_layer
        self.num_decoder_layer = num_decoder_layer
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learn_variance = learn_variance
        self.conditioning = label_embedder is not None
        # emb for label
        #   (batch, *) -> (batch, dim_base)
        self.label_embedder = label_embedder
        self.context_embedder = context_embedder

        if self.conditioning:
            if isinstance(label_embedder, CombinedEmbedder):
                self.num_label = len(label_embedder.dict_embedder)
            elif isinstance(label_embedder, ConcatEmbedder):
                self.num_label = len(label_embedder.list_embedder)
            else:
                raise ValueError(
                    "label_embedder must be CombinedEmbedder or ConcatEmbedder"
                )
        else:
            self.num_label = None

        self.dim_out = default(
            dim_out, num_in_channel * (1 if not learn_variance else 2)
        )
        # init projection
        self.init_proj = nn.Sequential(
            nn.Linear(num_in_channel, dim_base),
            nn.SiLU(),
            nn.Linear(dim_base, dim_base),
        )  # shape: (batch, sequence, num_in_channel) -> (batch, sequence, dim_base)

        # emb for time
        #   (batch, ) -> (batch, dim_base)
        self.time_embedder = TimeEmbedder(self.dim_base, self.type_pos_emb)

        # pos_emb for transformer
        #   use sinusoidal as the only option for now
        _transformer_pos_emb = SinusoidalPosEmb(self.dim_base)
        self.transformer_pos_emb = nn.Sequential(
            _transformer_pos_emb,
            nn.Linear(self.dim_base, self.dim_base),
            nn.GELU(),
            nn.Linear(self.dim_base, self.dim_base),
        )

        self.transformer = GPT2Model(
            dim_base=self.dim_base,
            num_attn_head=self.num_atten_head,
            num_layer=self.num_decoder_layer,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            conditioning=True,
            use_cross_attn=self.context_embedder is not None,
        )

        # adaLN modulation, for time or time+label
        #   init adaln modulation
        self.init_ln = nn.LayerNorm(self.dim_base, elementwise_affine=False, eps=1e-6)
        self.init_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim_base, self.dim_base * 2),
        )
        #   pre-final adaln modulation
        self.final_ln = nn.LayerNorm(self.dim_base, elementwise_affine=False, eps=1e-6)
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim_base, self.dim_base * 2),  # linear layer
        )

        # final linear
        self.final_linear = nn.Linear(self.dim_base * 2, self.dim_out)

        # initialize weights
        self.initialize_weights()

    def freeze_layers(self):
        "freeze all layers except init and final conv."
        self.time_mlp.requires_grad_(False)
        self.transformer_pos_emb.requires_grad_(False)
        self.init_proj.requires_grad_(True)  # True
        self.transformer.requires_grad_(False)
        if self.conditioning:
            self.final_ln.requires_grad_(False)
            self.final_adaLN_modulation.requires_grad_(False)
        for name, module in self.named_modules():
            # if the module name contains 'adaLN_modulation', then unfreeze it
            if "adaLN_modulation" in name:
                module.requires_grad_(True)
        self.final_linear.requires_grad_(True)  # True

    def initialize_weights(self):
        # initialize all transformer layers
        _weight_init_ignore_list = [nn.LayerNorm, nn.RMSNorm]

        def _basic_init(module):
            if isinstance(module, nn.Embedding):
                # initialize label embedder
                nn.init.normal_(module.weight, std=0.02)
            else:
                if getattr(module, "weight", None) is not None and not any(
                    isinstance(module, t) for t in _weight_init_ignore_list
                ):
                    nn.init.xavier_uniform_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # initialize time embedder
        nn.init.normal_(self.time_embedder.time_mlp[1].weight, std=0.02)
        nn.init.normal_(self.time_embedder.time_mlp[3].weight, std=0.02)

        # [option 1] zero-out final conv/linear layer
        # would this only be beneficial for epsilon prediction?
        # nn.init.constant_(self.final_linear.weight, 0)
        # nn.init.constant_(self.final_linear.bias, 0)

        # [option 2] initialize final linear layer with normal
        nn.init.normal_(self.final_linear.weight, std=0.02)
        nn.init.constant_(self.final_linear.bias, 0)

        if not self.conditioning:
            return  # normal LayerNorm is already initialized as gamma=1, beta=0

        # zero-out adaLN modulation layers in GPT2 blocks
        for decoder in self.transformer.decoders:
            # this layer: c -> 2 * (scale, shift,gate)
            nn.init.constant_(decoder.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(decoder.adaLN_modulation[-1].bias, 0)

        # zero-out final adaLN modulation layer
        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)

    @staticmethod
    def get_attn_mask(
        valid_length: Int[Tensor, "batch"],
        full_length: int,
        folded_length: int,
    ):
        """utility function to generate attention mask.

        Args:
            - valid_length: (batch, )
            - full_length: int, full length of the sequence
            - folded_length: int, folded length of the sequence

        Returns:
            - attn_mask: (batch, folded_length, folded_length)

        ---

        Example:
            get_attn_mask(
                valid_length=torch.tensor([5, 10, 13]),
                full_length=20,
                folded_length=10,
            )
            implied profile shape: [3, 10, 2]
            seq_mask [3, 10, 10]):
                [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
                ]
            attn_mask = seq_mask @ seq_mask.transpose(1, 2)
        """
        if full_length % folded_length != 0:
            raise ValueError(
                f"full_length {full_length} must be divisible by folded_length {folded_length}"
            )
        seq_mask = torch.arange(
            start=1, end=full_length + 1, device=valid_length.device
        )
        seq_mask = seq_mask.unsqueeze(0).expand(valid_length.shape[0], full_length)
        # shape: (batch, full_length)
        seq_mask = seq_mask <= valid_length.unsqueeze(1)  # shape: (batch, full_length)
        seq_mask = rearrange(
            seq_mask,
            "batch (folded channel) -> batch folded channel",
            folded=folded_length,
        )
        seq_mask = reduce(seq_mask, "batch folded channel -> batch folded", "max")
        attn_mask = einsum(
            seq_mask,
            seq_mask,
            "batch folded_a, batch folded_b -> batch folded_a folded_b",
        )  # shape: (batch, Lq, Lk)
        # for a given query, if no mask is True, then assign True to itself
        _all_ones = torch.ones_like(attn_mask)  # (b, Lq, Lk)
        _indices = ~attn_mask.any(dim=-1)  # (b, Lq)
        # _indices = einsum(_indices, _indices, "b Lq, b Lk -> b Lq Lk")
        _indices = torch.diag_embed(_indices, dim1=-2, dim2=-1)  # (b, Lq, Lk)
        attn_mask = torch.where(_indices, _all_ones, attn_mask)

        return attn_mask

    def _format_input_y(
        self,
        which: str,
        batch_size: int,
        device: torch.device,
        y: dict | None,
        dict_extras: dict | None,
    ):
        if which == "label":
            embedder = self.label_embedder
        elif which == "context":
            embedder = self.context_embedder
        else:
            raise ValueError("`which` must be label or context")
        assert embedder is not None, f"{which} embedder is None"

        force_drop_ids = torch.ones(
            (batch_size,), device=device, dtype=torch.long
        )  # shape: (batch, )

        # fill in missing keys
        if dict_extras is None:
            dict_extras = {}

        for label_name in embedder.embedder_keys:
            if label_name not in dict_extras:
                dict_extras[label_name] = {}

        if y is None:
            y = {}
            "if y is None, then seen as force drop all conditions"
        for label_name in getattr(embedder, "embedder_keys", []):
            if label_name not in y:
                y[label_name] = None
                dict_extras[label_name]["force_drop_ids"] = force_drop_ids

        return y, dict_extras

    def _check_start_pos(
        self,
        start_pos: Int[Tensor, "batch"],
        full_length: int,
        folded_length: int,
    ):
        """Check the value of start_pos.

        Assume full length L is folded into
        L = folded_length * folding_factor.
        `start_pos` should only be multiple of the folding_factor.

        Raises:
            ValueError: if start_pos is not multiple of folding_factor

        """
        factor = full_length // folded_length
        if torch.any(start_pos % factor != 0):
            raise ValueError(f"start_pos {start_pos} must be multiple of {factor}")

    def _get_folded_start_pos(
        self,
        start_pos: Int[Tensor, "batch"],
        full_length: int,
        folded_length: int,
    ):
        """Compute the start position after folding.

        start_pos_folded = start_pos // folding_factor

        """
        factor = full_length // folded_length
        return start_pos // factor

    def forward(
        self,
        x: Float[Tensor, "batch sequence num_in_channel"],
        t: Float[Tensor, "batch"],
        start_pos: int | Int[Tensor, "batch"] = 0,
        y: dict[str, Float[Tensor, "batch num_label"]] | None = None,
        c: dict[str, Float[Tensor, "batch num_label"]] | None = None,
        dict_emb_extras: dict[str, Any] | None = None,
        valid_length: Int[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch sequence num_in_channel"]:
        """DenoisingTransformer: Applies denoising to sequence data.

        Args:
            x: (batch, sequence, num_in_channel) data sequence
            t: (batch, ) time step
            start_pos: int, start position for sequence positional embedding,
                in FULL SEQUENCE unit
            y: dict[str, Tensor] | None, label conditions
            c: dict[str, Tensor] | None, label conditions, alias for y
            dict_emb_extras: extra arguments for label embedder
            valid_length: valid length for `x`, in FULL SEQUENCE unit. If None,
                uses full sequence length (no masking)

        Returns:
            x: (batch, sequence, num_in_channel) denoised sequence
        """
        batch_size, folded_length, num_in_channel = x.shape
        full_length = folded_length * num_in_channel
        # checks
        y = y or c  # c is alias for y
        if isinstance(start_pos, int):
            start_pos = torch.full(
                (x.shape[0],), start_pos, device=x.device, dtype=torch.int64
            )
        if start_pos.shape[0] != x.shape[0]:
            if x.shape[0] / start_pos.shape[0] != 2:
                raise ValueError(
                    f"start_pos shape {start_pos.shape} must match batch size {x.shape[0]} or be double the batch size"
                )
            start_pos = torch.cat([start_pos] * 2, dim=0)
        # Always create valid_length tensor - use full length if None provided
        if valid_length is None:
            valid_length = torch.full(
                (batch_size,), full_length, device=x.device, dtype=torch.long
            )
        elif x.shape[0] / valid_length.shape[0] not in [1, 2]:
            raise ValueError(
                f"valid_length shape {valid_length.shape} must match batch size {x.shape[0]} or be double the batch size"
            )
        elif valid_length.shape[0] != x.shape[0]:
            valid_length = torch.cat([valid_length] * 2, dim=0)

        # check start_pos (only in debug)
        # self._check_start_pos(
        #     start_pos,
        #     full_length=full_length,
        #     folded_length=folded_length,
        # )
        start_pos_folded = self._get_folded_start_pos(
            start_pos,
            full_length=full_length,
            folded_length=folded_length,
        )
        # --- step 1: project into model dmension ---
        # projection: x
        x = self.init_proj(x)
        # shape: (batch, sequence, num_in_channel) -> (batch, sequence, dim_base)
        # projection: t
        # generate embedding for time
        encoded_t = self.time_embedder(t)  # shape: (batch, dim_base)
        # [optional] projection: y as label
        if self.label_embedder is not None:
            _label_y, _dict_emb_extras = self._format_input_y(
                "label",
                batch_size,
                x.device,
                y,
                dict_emb_extras,
            )
            encoded_label = self.label_embedder(
                _label_y,
                dict_extra=_dict_emb_extras,
                batch_size=batch_size,
                device=x.device,
            )  # shape: (batch, dim_base)
        else:
            encoded_label = torch.zeros_like(encoded_t)  # shape: (batch, dim_base)
        encoded_ty = encoded_t + encoded_label  # shape: (batch, dim_base)
        encoded_ty = rearrange(encoded_ty, "batch dim_base -> batch 1 dim_base")
        # [optional] projection: y as context
        if self.context_embedder is not None:
            _context_y, _dict_emb_extras = self._format_input_y(
                "context",
                batch_size,
                x.device,
                y,
                dict_emb_extras,
            )
            encoded_context = self.context_embedder(
                _context_y, dict_extra=_dict_emb_extras
            )
            # shape: (batch, sequence, dim_base)
        else:
            encoded_context = None

        # init adaln modulation
        x_copy = x.clone()
        scale, shift = self.init_adaLN_modulation(encoded_ty).chunk(2, dim=-1)
        # shape: (batch, 1, dim_base), (batch, 1, dim_base)
        x = self.init_ln(x)  # shape: (batch, sequence, dim_base)
        x = x * (1 + scale) + shift  # shape: (batch, sequence, dim_base)
        x = F.silu(x)  # shape: (batch, sequence, dim_base)

        # transformer
        pos_seq = torch.arange(
            start=0, end=x.shape[1], device=x.device, dtype=x.dtype
        ).unsqueeze(0)  # shape: (1, sequence, )
        pos_seq = start_pos_folded.unsqueeze(1) + pos_seq  # shape: (batch, sequence)
        pos_emb_seq = self.transformer_pos_emb(
            pos_seq
        )  # shape: (batch, sequence, dim_base)
        x = x + pos_emb_seq  # sequence pos emb, shape: (batch, sequence, dim_base)

        # Always generate attention mask - no branching!
        attn_mask = self.get_attn_mask(
            valid_length=valid_length,
            full_length=folded_length * num_in_channel,
            folded_length=folded_length,
        ).to(x.device)  # shape: (batch, folded_length, folded_length)
        x, _ = self.transformer(
            x=x,
            y=encoded_ty,
            attn_mask=attn_mask,
            context=encoded_context,
            start_pos=start_pos_folded,
        )  # shape: (batch, sequence, channel)

        # final adaln modulation
        final_scale, final_shift = self.final_adaLN_modulation(encoded_ty).chunk(
            2, dim=-1
        )  # shape: (batch, 1, dim_base), (batch, 1, dim_base)
        x = self.final_ln(x)  # shape: (batch, sequence, dim_base)
        x = x * (1 + final_scale) + final_shift  # shape: (batch, sequence, dim_base)

        # final linear/conv layer
        # skip connection: discard in exp_id 2.1.0; readopted after 2.3.0 / 1.2.0
        x = torch.concat((x, x_copy), dim=-1)  # shape: (batch,  sequence, dim_base*2)
        x = self.final_linear(x)  # shape: (batch, sequence, dim_out)
        return x


class GPT2Block(nn.Module):
    """GPT2 block for 1D data

    Forward:
        input:
            - x: (batch, sequence, channel)

        return:
            - x: (batch, sequence, channel)

    """

    def __init__(
        self,
        dim_base: int,  # model dimension
        num_attn_head: int = 4,
        dim_head: None | int = None,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        conditioning: bool = False,
        use_cross_attn: bool = False,
    ):
        self.linear_attn = False
        super().__init__()
        self.dim_base = dim_base
        self.num_attn_head = num_attn_head
        self.dim_head = default(dim_head, dim_base // num_attn_head)
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conditioning = conditioning
        self.use_cross_attn = use_cross_attn
        if use_cross_attn and not conditioning:
            raise ValueError("cross attention must be used with conditioning.")

        if self.use_cross_attn:
            self.context_layers = nn.ModuleDict({})
            self.context_layers["ln_1"] = nn.LayerNorm(
                dim_base, elementwise_affine=not self.conditioning, eps=1e-6
            )
            self.context_layers["ln_2"] = nn.LayerNorm(
                dim_base, elementwise_affine=not self.conditioning, eps=1e-6
            )
            self.cross_attn = RMSNormSelfCrossAttention(
                dim_base,
                num_head=self.num_attn_head,
                use_context=self.use_cross_attn,
                linear=self.linear_attn,
            )
            self.context_layers["mlp"] = nn.Sequential(
                nn.Linear(dim_base, dim_feedforward),
                nn.SiLU(),
                nn.Linear(dim_feedforward, dim_base),
                nn.Dropout(self.dropout),
            )
            # assert conditioning
            self.context_layers["adaLN_modulation"] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_base, dim_base * 6),
            )
        else:
            self.attn = RMSNormSelfAttention(
                dim_base,
                num_head=self.num_attn_head,
                linear=self.linear_attn,
            )

        self.ln_1 = nn.LayerNorm(
            dim_base, elementwise_affine=not self.conditioning, eps=1e-6
        )
        self.ln_2 = nn.LayerNorm(
            dim_base, elementwise_affine=not self.conditioning, eps=1e-6
        )

        # no cross attention compared with the GPT2
        if not self.linear_attn:
            self.mlp = nn.Sequential(
                nn.Linear(dim_base, dim_feedforward),
                nn.SiLU(),
                nn.Linear(dim_feedforward, dim_base),
                nn.Dropout(self.dropout),
            )
        else:
            # use mixffn
            self.mlp = MixFFN(
                dim=dim_base,
            )

        # conditioning
        if self.conditioning:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_base, dim_base * 6),
            )

    def forward(
        self,
        x: Float[Tensor, "batch sequence channel"],
        y: None | Float[Tensor, "batch 1 channel"],
        attn_mask: None | Float[Tensor, "batch sequence sequence"],
        context: None | Float[Tensor, "batch seq2 channel"] = None,
        start_pos: int = 0,
    ) -> AttentionOutput:
        """
        Input:
            x - input tensor (batch, sequence, channel)
            y - label tensor (batch, 1, channel)
            attn_mask - attention mask (batch, sequence, sequence)
            context - context tensor (batch, seq2, channel)

        Return:
            x - updated sequence (batch, sequence, channel)
            if use_cross_attn:
            context - updated context sequence (batch, seq2, channel)
        """
        if self.conditioning and y is None:
            raise ValueError("y must be provided when conditioning is True.")
        if self.use_cross_attn and context is None:
            raise ValueError("context must be provided when use_cross_attn is True.")
        if self.conditioning:
            cond_scale_shift_gate = self.adaLN_modulation(
                y
            )  # shape: (batch, 1, channel * 6)
            scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = (
                cond_scale_shift_gate.chunk(6, dim=2)
            )
        else:
            scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
            ]
        if self.use_cross_attn:
            cond_scale_shift_gate = self.context_layers["adaLN_modulation"](
                y
            )  # shape: (batch, 1, channel * 6)
            (
                context_scale_attn,
                context_shift_attn,
                context_gate_attn,
                context_scale_mlp,
                context_shift_mlp,
                context_gate_mlp,
            ) = cond_scale_shift_gate.chunk(6, dim=2)

        # copy + normalization
        x_copy = x.clone()
        x = self.ln_1(x)
        x = x * (1.0 + scale_attn) + shift_attn  # scale shift from adaLN
        if self.use_cross_attn:
            context_copy = context.clone()
            context = self.context_layers["ln_1"](context)
            context = (
                context * (1.0 + context_scale_attn) + context_shift_attn
            )  # scale shift

        # attention
        if not self.use_cross_attn:
            attn_output = self.attn(
                x,
                attn_mask=attn_mask,
                start_pos=start_pos,
            )  # shape: (batch, sequence, channel)
            attn_x = attn_output.x_out
            attn_context = attn_output.context_out
        else:
            attn_output = self.cross_attn(
                x, context=context, attn_mask=attn_mask, start_pos=start_pos
            )
            attn_x = attn_output.x_out
            attn_context = attn_output.context_out
        x = x_copy + attn_x * gate_attn  # residual connection
        if self.use_cross_attn:
            context = context_copy + attn_context * context_gate_attn
            # residual connection

        # mlp / feedforward
        x_copy = x.clone()
        x = self.ln_2(x)
        x = x * (1.0 + scale_mlp) + shift_mlp  # scale shift from adaLN
        x = self.mlp(x)
        x = x_copy + x * gate_mlp  # residual connection
        if self.use_cross_attn:
            context_copy = context.clone()
            context = self.context_layers["ln_2"](context)
            context = context * (1.0 + context_scale_mlp) + context_shift_mlp
            context = self.context_layers["mlp"](context)
            context = context_copy + context * context_gate_mlp  # residual
            return AttentionOutput(x, context)  # (b, seq, c), (b, seq2, c)

        return AttentionOutput(x)  # shape: (batch, sequence, channel)


class GPT2Model(nn.Module):
    """base GPT2 model

    Forward:
        input:
            - x: (batch, sequence, channel)
            - y: None|(batch, 1, channel)
            - context: None|(batch, seq2, channel)
            - attn_mask: None|(batch, sequence, sequence)

        return:
            - x: (batch, sequence, channel)
            - (if use_cross_attn):
            - context: (batch, seq2, channel)
    """

    def __init__(
        self,
        dim_base: int,
        num_attn_head: int = 4,
        num_layer=6,
        dim_feedforward=2048,  # why so big by default?
        dropout=0.1,
        conditioning: bool = False,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        self.dim_base = dim_base
        self.num_attn_head = num_attn_head
        self.num_layer = num_layer
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conditioning = conditioning
        self.use_cross_attn = use_cross_attn

        # define model layers here
        self.decoders = nn.ModuleList([])
        for _idx in range(self.num_layer):
            self.decoders.append(
                GPT2Block(
                    dim_base=self.dim_base,
                    num_attn_head=self.num_attn_head,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    conditioning=self.conditioning,
                    use_cross_attn=self.use_cross_attn,
                )  # shape: (batch, dim_base, channel) -> (batch, dim_base, channel)
            )

    def forward(
        self,
        x: Float[Tensor, "batch sequence channel"],
        y: None | Float[Tensor, "batch sequence channel"] = None,
        attn_mask: None | Float[Tensor, "batch sequence sequence"] = None,
        context: None | Float[Tensor, "batch seq2 channel"] = None,
        start_pos: int = 0,
    ) -> AttentionOutput:
        """
        Input:
            x - input tensor (batch, sequence, channel)
            y - label tensor (batch, 1, channel)
            attn_mask - attention mask (batch, sequence, sequence)
            context - context tensor (batch, seq2, channel)
            start_pos - int, start position for sequence positional embedding

        Return:
            x - updated sequence (batch, sequence, channel)
            if use_cross_attn:
            context - updated context sequence (batch, seq2, channel)
        """
        structured_inout = AttentionOutput(x_out=x, context_out=context)
        for decoder in self.decoders:
            structured_inout = decoder(
                x=structured_inout.x_out,
                y=y,
                attn_mask=attn_mask,
                context=structured_inout.context_out,
                start_pos=start_pos,
            )
        return structured_inout  # shape: (B, L, C), None| (B, L2, C)


class MLPBlock(nn.Module):
    "layernorm -> linear -> silu -> linear (dropout) -> (gate)"

    def __init__(
        self,
        dim_base: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        conditioning: bool = False,
    ):
        super().__init__()
        self.dim_base = dim_base
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conditioning = conditioning

        self.ln_1 = nn.LayerNorm(
            dim_base, elementwise_affine=not self.conditioning, eps=1e-6
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_base, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, dim_base),
            nn.Dropout(self.dropout),
        )

        # conditioning
        if self.conditioning:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_base, dim_base * 3),
            )

    def forward(
        self,
        x: Float[Tensor, "batch channel"],
        c: None | Float[Tensor, "batch channel"] = None,
    ) -> Float[Tensor, "batch channel"]:
        if c is not None and self.conditioning:
            cond_scale_shift_gate = self.adaLN_modulation(
                c
            )  # shape: (batch, channel * 3)
            scale, shift, gate = cond_scale_shift_gate.chunk(3, dim=1)
        else:
            scale, shift, gate = [0.0, 0.0, 1.0]  # no conditioning

        x_copy = (
            x.clone()
        )  # can also just assign, since we do not use in-place operation
        x = self.ln_1(x)
        x = x * (1.0 + scale) + shift  # scale shift from adaLN
        x = self.mlp(x)
        x = x_copy + x * gate  # residual connection

        return x


class DenoisingMLP1D(nn.Module):
    """denoising MLP for 1D data

    Arguments:
        - dim_base: d_model of transformer, recommended value = ?
        - ...
        - conditioning: whether to use conditioning,
        the condition tensor 'c' should have same dimension as `dim_base`
    Forward:
        input:
            - x: (batch, num_in_channel, sequence)
            - time: (batch, )
            - c: None|(batch, num_cond_channel, 1)
            - x_self_cond: None|(batch, num_in_channel, sequence)
            - cfg_scale: float

        return:
            - x: (batch, dim_out, sequence)
    """  # noqa: E501

    def __init__(
        self,
        dim_base: int,
        seq_length: int,
        num_in_channel: int = 3,
        self_condition: bool = False,
        dim_feedforward: int = 2048,
        num_block: int = 6,
        type_pos_emb: str = "sinusoidal",
        dropout=0.1,
        learn_variance: bool = False,
        conditioning: bool = False,
        cond_embedder: EmbedderWrapper | None = None,
    ):
        super().__init__()
        self.dim_base = dim_base
        self.seq_length = seq_length
        self.num_in_channel = num_in_channel
        self.self_condition = self_condition
        self.dim_feedforward = dim_feedforward
        assert num_block >= 1, "num_layer must be >= 1"
        self.num_block = num_block
        self.type_pos_emb = type_pos_emb
        self.dropout = dropout
        self.learn_variance = learn_variance
        self.conditioning = conditioning
        self.cond_embedder = cond_embedder
        # (batch, num_condition_channel, sequence)
        #   -> (batch, dim_base, sequence)
        self.dim_out = num_in_channel * (1 if not learn_variance else 2)

        Linear_Op = partial(nn.Conv1d, kernel_size=1, padding=0, stride=1, bias=True)

        if self.type_pos_emb == "learned":
            pos_emb = LearnedSinusoidalPosEmb(self.dim_base)
        elif self.type_pos_emb == "random":
            pos_emb = RandomSinusoidalPosEmb(self.dim_base)
        else:
            pos_emb = SinusoidalPosEmb(self.dim_base)
        self.time_mlp = nn.Sequential(
            pos_emb,  # (batch, ) -> (batch, dim_base)
            nn.Linear(self.dim_base, self.dim_base),
            nn.GELU(),
            nn.Linear(self.dim_base, self.dim_base * 2),
        )

        self.init_proj = Linear_Op(
            (
                seq_length * num_in_channel
                if not self.self_condition
                else num_in_channel * 2
            ),
            dim_base,
        )
        self.final_proj = Linear_Op(dim_base * 2, seq_length * self.dim_out)

        self.hidden_mlp = nn.ModuleList([])
        for _idx in range(0, self.num_block):
            self.hidden_mlp.append(
                MLPBlock(
                    dim_base=self.dim_base,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    conditioning=self.conditioning,
                )
            )

    def forward(
        self,
        x: Float[Tensor, "batch num_in_channel sequence"],
        time: Float[Tensor, "batch"],
        c: Float[Tensor, "batch num_cond_channel 1"] | None = None,
        force_drop_ids: Float[Tensor, "batch"] | None = None,
        x_self_cond: Float[Tensor, ""] | None = None,
    ) -> Float[Tensor, ""]:
        # reshape for MLP
        # num_in_channel = x.shape[1]
        # mid_dim = x.shape[1] // 2 # eliminate vectorize
        # x = x[:, mid_dim:mid_dim+1, :] # shape: (batch, 1, sequence)
        x = rearrange(x, "batch channel sequence -> batch (channel sequence) 1")

        # encode timestep
        encoded_t = self.time_mlp(time)  # shape: (batch, dim_base * 2)
        encoded_t = rearrange(
            encoded_t, "b d -> b d 1"
        )  # shape: (batch, dim_base * 2, 1)
        scale, shift = encoded_t.chunk(
            2, dim=1
        )  # shape: (batch, dim_base, 1), (batch, dim_base, 1)

        # encode condition
        if self.conditioning and c is not None:
            encoded_c = self.cond_embedder(c, force_drop_ids=force_drop_ids)
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
            encoded_c = None

        # init proj
        if self.self_condition:
            if x_self_cond is None:
                warnings.warn(
                    "self_condition is True, but x_self_cond is None, \
                        using full zeros.",
                    stacklevel=2,
                )
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim=1)
        x = self.init_proj(x)  # shape: (batch, dim_base, 1)
        x_copy = x.clone()
        x = x * (1 + scale) + shift
        x = F.silu(x)  # shape: (batch, dim_base, 1)

        # hidden mlp layers
        x = rearrange(
            x, "batch channel sequence -> batch (channel sequence)"
        )  # shape: (batch, dim_base)
        encoded_c = (
            rearrange(encoded_c, "batch channel sequence -> batch (channel sequence)")
            if encoded_c is not None
            else None
        )
        for hidden_mlp in self.hidden_mlp:
            x = hidden_mlp(x, encoded_c)  # shape: (batch, dim_base)

        x = rearrange(
            x,
            "batch (channel sequence) -> batch channel sequence",
            channel=self.dim_base,
        )  # shape: (batch, dim_base, 1)
        encoded_c = (
            rearrange(
                encoded_c,
                "batch (channel sequence) -> batch channel sequence",
                channel=self.dim_base,
            )
            if encoded_c is not None
            else None
        )

        # skipping the final adaLN_modulation
        # final proj
        x = torch.concat([x, x_copy], dim=1)  # shape: (batch, dim_base * 2, 1)
        x = self.final_proj(x)  # shape: (batch, dim_out, 1)
        x = rearrange(
            x,
            "batch (channel sequence) 1 -> batch channel sequence",
            channel=self.dim_out,
        )  # shape: (batch, dim_out, 1)

        return x  # shape: (batch, dim_out, 1)

    def forward_with_cfg(
        self,
        x: Float[Tensor, "batch num_in_channel sequence"],
        time: Float[Tensor, "batch"],
        c: Float[Tensor, "batch num_cond_channel 1"] | None = None,
        x_self_cond: Float[Tensor, ""] | None = None,
        cfg_scale: float = 1.0,
    ):
        if c is None or cfg_scale == 1.0:
            return self.forward(x, time, c, x_self_cond)

        _x = torch.cat([x, x], dim=0)
        _time = torch.cat([time, time], dim=0)
        _c = torch.cat([c, c], dim=0)
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
        cond_x, uncond_x = self.forward(
            _x, _time, _c, force_drop_ids=force_drop_ids, x_self_cond=_x_self_cond
        ).chunk(2, dim=0)
        scaled_x = uncond_x + cfg_scale * (cond_x - uncond_x)

        return scaled_x


def get_start_pos(
    first_day_of_week: int | Int[Tensor, "batch"],
    steps_per_day: int,
) -> int | Int[Tensor, "batch"]:
    """get `start_pos` for monthly sequence data.

    Args:
        first_day_of_week: int, first day of week, 0-6
        steps_per_day: int, steps per day

    Returns:
        int: start_pos

    Examples:
        >>> get_start_pos(0, 24)
        0
        >>> get_start_pos(1, 24)
        24
        >>> get_start_pos(2, 24)
        48

    """
    if isinstance(first_day_of_week, int):
        first_day_of_week = torch.tensor(first_day_of_week, dtype=torch.int64)
    if torch.any(first_day_of_week < 0) or torch.any(first_day_of_week > 6):
        raise ValueError("first_day_of_week must be in [0, 6]")
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be > 0")
    start_pos = first_day_of_week * steps_per_day
    if isinstance(start_pos, int):
        start_pos = start_pos.item()
    return start_pos
