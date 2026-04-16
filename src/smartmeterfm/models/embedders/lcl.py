"""LCL (Low Carbon London) data embedders for conditional generation.

Provides embedders for the LCL smart meter electricity dataset (Zenodo).
The primary embedder embeds both month (0-11) and year (raw calendar year,
offset-adjusted internally to 0-indexed) for conditional generation.
"""

from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from ..nn_components import CombinedEmbedder, IntegerEmbedder
from ._registry import register_embedder


@register_embedder("lcl_month_year")
class LCLMonthYearEmbedder(CombinedEmbedder):
    """Month and year embedder for LCL electricity data.

    Embeds month (0-11) and year.  Year values in the condition dict are
    raw calendar years (e.g. 2012, 2013); the embedder subtracts
    *year_offset* before indexing into the year embedding table.

    Args:
        month: Dictionary with IntegerEmbedder arguments for month embedding.
        year: Dictionary with IntegerEmbedder arguments for year embedding.
            ``num_embedding`` defaults to 2 (covering 2012-2013).
        year_offset: Value subtracted from raw year before embedding lookup.
            Default 2012 maps 2012→0, 2013→1.
    """

    def __init__(
        self,
        month: dict[str, Any],
        year: dict[str, Any],
        year_offset: int = 2012,
    ):
        if "num_embedding" not in month:
            month["num_embedding"] = 12
        if "num_embedding" not in year:
            year["num_embedding"] = 2
        super().__init__(
            {
                "month": IntegerEmbedder(**month),
                "year": IntegerEmbedder(**year),
            }
        )
        self.year_offset = year_offset

    def forward(
        self,
        dict_labels: dict[str, Float[Tensor, "batch"]] | None,
        dict_extra: dict[str, Any] | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> Float[Tensor, "batch dim_base"]:
        # Apply year offset so raw calendar years become 0-indexed
        if dict_labels is not None and "year" in dict_labels:
            dict_labels = dict(dict_labels)  # shallow copy to avoid mutation
            dict_labels["year"] = dict_labels["year"] - self.year_offset
        return super().forward(dict_labels, dict_extra, batch_size, device)
