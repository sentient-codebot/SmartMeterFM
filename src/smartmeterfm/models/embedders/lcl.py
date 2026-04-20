"""LCL (Low Carbon London) data embedders for conditional generation.

Provides embedders for the LCL smart meter electricity dataset (Zenodo).
Both embedders embed four condition fields:

    * ``month`` (0-11)              — integer embedding
    * ``year``                      — position (sinusoidal) embedding, with a
      configurable ``year_offset`` so raw calendar years (e.g. 2012, 2013) are
      internally shifted to a 0-indexed range before lookup
    * ``first_day_of_week`` (0-6)   — integer embedding
    * ``month_length`` (0-3)        — integer embedding, i.e. ``num_days - 28``

``first_day_of_week`` and ``month_length`` are calendar fields derivable from
``(year, month)`` via :func:`calendar.monthrange`; they are auto-derived by
:class:`smartmeterfm.conditions.LCLCondition` at sampling time and by the
training-script collate function (``make_collate_fn`` in ``train_flow.py``).

Two flavours are registered:

    * ``lcl_label``   — sums the four embeddings into a single ``[batch, dim]``
      vector (for AdaLN conditioning).
    * ``lcl_context`` — stacks them into a ``[batch, seq, dim]`` sequence (for
      cross-attention conditioning).
"""

from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from ..nn_components import (
    CombinedEmbedder,
    ContextEmbedder,
    IntegerEmbedder,
    PositionEmbedder,
)
from ._registry import register_embedder


def _apply_year_offset(
    dict_labels: dict[str, Tensor] | None, year_offset: int
) -> dict[str, Tensor] | None:
    """Shift raw calendar years to 0-indexed values (shallow-copy on mutation)."""
    if dict_labels is None or "year" not in dict_labels:
        return dict_labels
    if dict_labels["year"] is None:
        return dict_labels
    dict_labels = dict(dict_labels)
    dict_labels["year"] = dict_labels["year"] - year_offset
    return dict_labels


@register_embedder("lcl_label")
class LCLLabelEmbedder(CombinedEmbedder):
    """Label (sum) embedder for LCL electricity data.

    Embeds ``month``, ``year``, ``first_day_of_week`` and ``month_length`` and
    sums their embeddings into a single ``[batch, dim_embedding]`` vector
    suitable for AdaLN conditioning.

    Args:
        month: :class:`IntegerEmbedder` kwargs for month (0-11).
            ``num_embedding`` defaults to 12.
        year: :class:`PositionEmbedder` kwargs for year. Only accepts
            ``dim_embedding`` and ``dropout``; year is embedded via a
            sinusoidal position embedding + MLP rather than a lookup table.
        first_day_of_week: :class:`IntegerEmbedder` kwargs for the weekday of
            the first day of the month (0-6).
        month_length: :class:`IntegerEmbedder` kwargs for ``num_days - 28``
            (0-3, covering 28-31 day months).
        year_offset: Value subtracted from raw year before the position
            embedding lookup. Default ``2012`` maps 2012→0, 2013→1.
    """

    def __init__(
        self,
        month: dict[str, Any],
        year: dict[str, Any],
        first_day_of_week: dict[str, Any],
        month_length: dict[str, Any],
        year_offset: int = 2012,
    ):
        if "num_embedding" not in month:
            month["num_embedding"] = 12
        super().__init__(
            {
                "month": IntegerEmbedder(**month),
                "year": PositionEmbedder(**year),
                "first_day_of_week": IntegerEmbedder(**first_day_of_week),
                "month_length": IntegerEmbedder(**month_length),
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
        dict_labels = _apply_year_offset(dict_labels, self.year_offset)
        return super().forward(dict_labels, dict_extra, batch_size, device)


@register_embedder("lcl_context")
class LCLContextEmbedder(ContextEmbedder):
    """Context (sequence) embedder for LCL electricity data.

    Same four fields as :class:`LCLLabelEmbedder` (month, year,
    first_day_of_week, month_length), but the per-field embeddings are
    concatenated along a sequence dimension to produce a
    ``[batch, seq, dim_embedding]`` tensor suitable for cross-attention
    conditioning.

    Args match :class:`LCLLabelEmbedder`.
    """

    def __init__(
        self,
        month: dict[str, Any],
        year: dict[str, Any],
        first_day_of_week: dict[str, Any],
        month_length: dict[str, Any],
        year_offset: int = 2012,
    ):
        if "num_embedding" not in month:
            month["num_embedding"] = 12
        super().__init__(
            {
                "month": IntegerEmbedder(**month),
                "year": PositionEmbedder(**year),
                "first_day_of_week": IntegerEmbedder(**first_day_of_week),
                "month_length": IntegerEmbedder(**month_length),
            }
        )
        self.year_offset = year_offset

    def forward(
        self,
        dict_labels: dict[str, Float[Tensor, "batch"]] | None,
        dict_extra: dict[str, Any] | None = None,
    ) -> Float[Tensor, "batch seq dim_base"]:
        dict_labels = _apply_year_offset(dict_labels, self.year_offset)
        return super().forward(dict_labels, dict_extra)
