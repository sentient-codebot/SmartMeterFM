"""Sample condition management for conditional generation.

Provides typed, validated condition objects that convert to model-ready
``dict[str, Tensor]`` via a generic ``to_tensor_dict()`` method.  Adding a new
condition field requires only an ``Annotated`` type hint with
``ConditionFieldMeta`` -- no per-field boilerplate.

Extensibility
-------------
Adding a field to an existing condition class::

    class LCLCondition(SampleCondition):
        month: Annotated[int | None, ConditionFieldMeta()] = None
        tariff_type: Annotated[int | None, ConditionFieldMeta()] = None  # new

``to_tensor_dict()`` will automatically include ``"tariff_type"`` when set.
The model's embedder must also have a matching key in its ``dict_embedder``.

Adding a derived field (auto-computed from another)::

    is_winter: Annotated[int | None, ConditionFieldMeta(
        derive_from="month", derive_fn=lambda m: 1 if m in (0, 1, 11) else 0
    )] = None

Adding a new dataset condition class:

1. Subclass ``SampleCondition``
2. Declare fields with ``Annotated[type, ConditionFieldMeta(...)]``
3. Add a ``_validate_ranges`` model validator for domain constraints
4. Use it in the sampling callback / generation script

Checklist for new conditions:

1. Add annotated field to the appropriate condition class
2. Ensure the model's embedder has a matching key (in ``embedders/`` registry)
3. Ensure the data module produces labels with the same key (in ``data_modules/``)
4. Update TOML config if the condition should be controllable at config time
"""

from __future__ import annotations

import calendar
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

import torch
from pydantic import BaseModel, ConfigDict, model_validator


@dataclass(frozen=True)
class ConditionFieldMeta:
    """Metadata attached to a condition field via ``Annotated[..., ConditionFieldMeta(...)]``.

    Attributes:
        dtype: Tensor dtype for this field (default: ``torch.long``).
        offset: Value subtracted before tensor creation (e.g. if raw data is
            1-indexed but the embedder expects 0-indexed).
        derive_from: Name of another field to auto-derive this value from.
        derive_fn: Function ``parent_value -> this_value`` used when
            ``derive_from`` is set and this field is ``None``.
    """

    dtype: torch.dtype = torch.long
    offset: int = 0
    derive_from: str | None = None
    derive_fn: Callable[[int], int] | None = None


class SampleCondition(BaseModel):
    """Base class for sample conditions.

    Subclasses declare fields with ``Annotated[int | None, ConditionFieldMeta(...)]``.
    The generic ``to_tensor_dict()`` and ``to_force_drop_ids()`` methods iterate
    over these fields automatically.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _get_field_metas(cls) -> dict[str, ConditionFieldMeta]:
        """Extract ``ConditionFieldMeta`` from Pydantic ``model_fields`` metadata."""
        result: dict[str, ConditionFieldMeta] = {}
        for name, info in cls.model_fields.items():
            for meta in info.metadata:
                if isinstance(meta, ConditionFieldMeta):
                    result[name] = meta
                    break
        return result

    @model_validator(mode="after")
    def _derive_fields(self) -> SampleCondition:
        """Auto-populate derived fields from their parent fields."""
        for name, meta in self._get_field_metas().items():
            if (
                meta.derive_from is not None
                and meta.derive_fn is not None
                and getattr(self, name) is None
            ):
                parent_val = getattr(self, meta.derive_from)
                if parent_val is not None:
                    object.__setattr__(self, name, meta.derive_fn(parent_val))
        return self

    def to_tensor_dict(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Convert to model-ready condition dict.

        Only includes fields that are not ``None``.  Each tensor has shape
        ``(batch_size, 1)`` with the value offset-adjusted.
        """
        result: dict[str, torch.Tensor] = {}
        for name, meta in self._get_field_metas().items():
            value = getattr(self, name)
            if value is not None:
                result[name] = torch.full(
                    (batch_size, 1),
                    value - meta.offset,
                    dtype=meta.dtype,
                    device=device,
                )
        return result

    def to_force_drop_ids(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Per-field ``force_drop_ids`` for classifier-free guidance.

        Fields set to ``None`` get ``1`` (unconditional / drop), set fields
        get ``0`` (keep).
        """
        result: dict[str, torch.Tensor] = {}
        for name in self._get_field_metas():
            drop = 1 if getattr(self, name) is None else 0
            result[name] = torch.full(
                (batch_size,), drop, dtype=torch.long, device=device
            )
        return result


# ---------------------------------------------------------------------------
# Month-to-season mapping (matches embedder buffer in models/embedders/wpuq.py)
# ---------------------------------------------------------------------------

_MONTH_TO_SEASON = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]


def _month_to_season(month: int) -> int:
    if not (0 <= month < len(_MONTH_TO_SEASON)):
        raise ValueError(f"month must be 0-11, got {month}")
    return _MONTH_TO_SEASON[month]


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


def _derive_calendar_from_year_month(year: int, month: int) -> tuple[int, int]:
    """Return ``(first_day_of_week, month_length_offset)`` for *year* / *month*.

    *month* is 0-indexed (0 = January).  ``month_length_offset`` is
    ``num_days - 28`` so it fits into the 0-3 range used by the embedders.
    """
    weekday, days = calendar.monthrange(year, month + 1)
    return weekday, days - 28


# ---------------------------------------------------------------------------
# Dataset-specific condition classes
# ---------------------------------------------------------------------------


class WPuQCondition(SampleCondition):
    """Sample condition for WPuQ heat-pump / household data.

    All fields are optional (``None`` = unconditional for that field).
    Month uses 0-indexed values (0 = January, 11 = December).
    Season is auto-derived from month when not set explicitly.

    For monthly-segmented data, ``year`` is stored alongside ``month`` so
    that ``first_day_of_week`` and ``month_length`` can be derived via
    ``calendar.monthrange(year, month+1)``.
    """

    month: Annotated[int | None, ConditionFieldMeta()] = None
    season: Annotated[
        int | None,
        ConditionFieldMeta(derive_from="month", derive_fn=_month_to_season),
    ] = None
    day_type: Annotated[int | None, ConditionFieldMeta()] = None
    household_id: Annotated[int | None, ConditionFieldMeta()] = None
    year: Annotated[int | None, ConditionFieldMeta()] = None
    first_day_of_week: Annotated[int | None, ConditionFieldMeta()] = None  # 0-6
    month_length: Annotated[int | None, ConditionFieldMeta()] = None  # 0-3

    @model_validator(mode="after")
    def _derive_calendar_fields(self) -> WPuQCondition:
        if self.year is not None and self.month is not None:
            weekday, ml = _derive_calendar_from_year_month(self.year, self.month)
            if self.first_day_of_week is None:
                object.__setattr__(self, "first_day_of_week", weekday)
            if self.month_length is None:
                object.__setattr__(self, "month_length", ml)
        return self

    @model_validator(mode="after")
    def _validate_ranges(self) -> WPuQCondition:
        if self.month is not None and not (0 <= self.month <= 11):
            raise ValueError(f"month must be 0-11, got {self.month}")
        if self.season is not None and not (0 <= self.season <= 3):
            raise ValueError(f"season must be 0-3, got {self.season}")
        if self.day_type is not None and not (0 <= self.day_type <= 1):
            raise ValueError(f"day_type must be 0 or 1, got {self.day_type}")
        if self.first_day_of_week is not None and not (
            0 <= self.first_day_of_week <= 6
        ):
            raise ValueError(
                f"first_day_of_week must be 0-6, got {self.first_day_of_week}"
            )
        if self.month_length is not None and not (0 <= self.month_length <= 3):
            raise ValueError(f"month_length must be 0-3, got {self.month_length}")
        return self


class LCLCondition(SampleCondition):
    """Sample condition for LCL (London) electricity data.

    Currently month-only, matching the LCL data module.  Future LCL-specific
    conditions (tariff type, Acorn group, etc.) can be added here.

    For monthly-segmented data, ``year`` is stored alongside ``month`` so
    that ``first_day_of_week`` and ``month_length`` can be derived via
    ``calendar.monthrange(year, month+1)``.
    """

    month: Annotated[int | None, ConditionFieldMeta()] = None
    year: Annotated[int | None, ConditionFieldMeta()] = None
    first_day_of_week: Annotated[int | None, ConditionFieldMeta()] = None  # 0-6
    month_length: Annotated[int | None, ConditionFieldMeta()] = None  # 0-3

    @model_validator(mode="after")
    def _derive_calendar_fields(self) -> LCLCondition:
        if self.year is not None and self.month is not None:
            weekday, ml = _derive_calendar_from_year_month(self.year, self.month)
            if self.first_day_of_week is None:
                object.__setattr__(self, "first_day_of_week", weekday)
            if self.month_length is None:
                object.__setattr__(self, "month_length", ml)
        return self

    @model_validator(mode="after")
    def _validate_ranges(self) -> LCLCondition:
        if self.month is not None and not (0 <= self.month <= 11):
            raise ValueError(f"month must be 0-11, got {self.month}")
        if self.first_day_of_week is not None and not (
            0 <= self.first_day_of_week <= 6
        ):
            raise ValueError(
                f"first_day_of_week must be 0-6, got {self.first_day_of_week}"
            )
        if self.month_length is not None and not (0 <= self.month_length <= 3):
            raise ValueError(f"month_length must be 0-3, got {self.month_length}")
        return self
