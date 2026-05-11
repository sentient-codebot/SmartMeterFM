"""LCL (Low Carbon London) data embedders for conditional generation.

Provides embedders for the LCL smart meter electricity dataset.  Both
embedders embed four required calendar fields plus two optional per-household
attributes:

    * ``month`` (0-11)              — integer embedding
    * ``year``                      — position (sinusoidal) embedding, with a
      configurable ``year_offset`` so raw calendar years (e.g. 2012, 2013) are
      internally shifted to a 0-indexed range before lookup
    * ``first_day_of_week`` (0-6)   — integer embedding
    * ``month_length`` (0-3)        — integer embedding, i.e. ``num_days - 28``
    * ``tariff_type`` (0-1, optional)      — integer embedding for Std/ToU
    * ``acorn_grouped`` (0-3, optional)    — integer embedding for ACORN class
      (Affluent / Comfortable / Adversity / Unclassified)

``first_day_of_week`` and ``month_length`` are calendar fields derivable from
``(year, month)`` via :func:`calendar.monthrange`; they are auto-derived by
:class:`smartmeterfm.conditions.LCLCondition` at sampling time and by the
training-script collate function (``make_collate_fn`` in ``train_flow.py``).

``tariff_type`` and ``acorn_grouped`` are *optional*: when their constructor
kwargs are ``None`` (the default), the embeddings are not added to the
underlying :class:`~smartmeterfm.models.nn_components.CombinedEmbedder` /
:class:`~smartmeterfm.models.nn_components.ContextEmbedder` ``ModuleDict``,
preserving ``state_dict`` keys for checkpoints trained without these fields.

Two flavours are registered:

    * ``lcl_label``   — sums the embeddings into a single ``[batch, dim]``
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


def _build_lcl_embedder_dict(
    month: dict[str, Any],
    year: dict[str, Any],
    first_day_of_week: dict[str, Any],
    month_length: dict[str, Any],
    tariff_type: dict[str, Any] | None,
    acorn_grouped: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the ``dict_embedder`` for both LCL embedder flavours.

    Required calendar fields are always present.  Optional per-household
    fields are added only when the caller passes a non-None kwarg dict, so
    omitting them yields the same ``state_dict`` keys as the pre-tariff/acorn
    embedder did.
    """
    if "num_embedding" not in month:
        month["num_embedding"] = 12
    embedders: dict[str, Any] = {
        "month": IntegerEmbedder(**month),
        "year": PositionEmbedder(**year),
        "first_day_of_week": IntegerEmbedder(**first_day_of_week),
        "month_length": IntegerEmbedder(**month_length),
    }
    if tariff_type is not None:
        if "num_embedding" not in tariff_type:
            tariff_type["num_embedding"] = 2
        embedders["tariff_type"] = IntegerEmbedder(**tariff_type)
    if acorn_grouped is not None:
        if "num_embedding" not in acorn_grouped:
            acorn_grouped["num_embedding"] = 4
        embedders["acorn_grouped"] = IntegerEmbedder(**acorn_grouped)
    return embedders


@register_embedder("lcl_label")
class LCLLabelEmbedder(CombinedEmbedder):
    """Label (sum) embedder for LCL electricity data.

    Embeds the four required calendar fields (``month``, ``year``,
    ``first_day_of_week``, ``month_length``) plus, optionally,
    ``tariff_type`` and ``acorn_grouped`` per-household attributes, summing
    them into a single ``[batch, dim_embedding]`` vector suitable for AdaLN
    conditioning.

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
        tariff_type: Optional :class:`IntegerEmbedder` kwargs for Std/ToU
            tariff (0-1).  Default ``num_embedding=2``.  Pass ``None`` (the
            default) to omit this field — preserves backward-compat
            ``state_dict`` keys.
        acorn_grouped: Optional :class:`IntegerEmbedder` kwargs for ACORN
            grouped class (0-3, Affluent/Comfortable/Adversity/Unclassified).
            Default ``num_embedding=4``.  Pass ``None`` (the default) to omit.
        year_offset: Value subtracted from raw year before the position
            embedding lookup. Default ``2012`` maps 2012→0, 2013→1.
    """

    def __init__(
        self,
        month: dict[str, Any],
        year: dict[str, Any],
        first_day_of_week: dict[str, Any],
        month_length: dict[str, Any],
        tariff_type: dict[str, Any] | None = None,
        acorn_grouped: dict[str, Any] | None = None,
        year_offset: int = 2012,
    ):
        super().__init__(
            _build_lcl_embedder_dict(
                month=month,
                year=year,
                first_day_of_week=first_day_of_week,
                month_length=month_length,
                tariff_type=tariff_type,
                acorn_grouped=acorn_grouped,
            )
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

    Same fields as :class:`LCLLabelEmbedder` (month, year, first_day_of_week,
    month_length, plus optional tariff_type / acorn_grouped), but the
    per-field embeddings are concatenated along a sequence dimension to
    produce a ``[batch, seq, dim_embedding]`` tensor suitable for
    cross-attention conditioning.

    Args match :class:`LCLLabelEmbedder`.
    """

    def __init__(
        self,
        month: dict[str, Any],
        year: dict[str, Any],
        first_day_of_week: dict[str, Any],
        month_length: dict[str, Any],
        tariff_type: dict[str, Any] | None = None,
        acorn_grouped: dict[str, Any] | None = None,
        year_offset: int = 2012,
    ):
        super().__init__(
            _build_lcl_embedder_dict(
                month=month,
                year=year,
                first_day_of_week=first_day_of_week,
                month_length=month_length,
                tariff_type=tariff_type,
                acorn_grouped=acorn_grouped,
            )
        )
        self.year_offset = year_offset

    def forward(
        self,
        dict_labels: dict[str, Float[Tensor, "batch"]] | None,
        dict_extra: dict[str, Any] | None = None,
    ) -> Float[Tensor, "batch seq dim_base"]:
        dict_labels = _apply_year_offset(dict_labels, self.year_offset)
        return super().forward(dict_labels, dict_extra)
