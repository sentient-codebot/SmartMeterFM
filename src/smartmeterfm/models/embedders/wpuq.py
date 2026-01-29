"""WPuQ (Heat Pump) data embedders for conditional generation.

This module provides embedders for the publicly available WPuQ dataset,
which contains heat pump electricity consumption data from Germany.

These embedders are designed to work without any proprietary dependencies
and serve as a showcase for the SmartMeterFM library.
"""

from typing import Any

import torch
import torch.nn as nn

from ._registry import register_embedder
from ..nn_components import CombinedEmbedder, IntegerEmbedder


@register_embedder("wpuq_month")
class WPuQMonthEmbedder(nn.Module):
    """Simple month-based embedder for WPuQ data.

    Embeds month information (0-11) for conditional generation.
    Supports classifier-free guidance through dropout.

    Args:
        month: Dictionary with IntegerEmbedder arguments for month embedding.
            Required keys: num_embedding (default: 12), dim_embedding
    """

    def __init__(self, month: dict[str, Any]):
        super().__init__()

        # Set default num_embedding for months if not provided
        if "num_embedding" not in month:
            month["num_embedding"] = 12

        self.month_embedder = IntegerEmbedder(**month)
        self.dim_out = month["dim_embedding"]

    def forward(
        self,
        y: dict[str, torch.Tensor],
        train: bool | None = None,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed month condition.

        Args:
            y: Dictionary containing 'month' tensor of shape [batch, 1] or [batch]
            train: Whether in training mode (for dropout). None uses self.training
            force_drop_ids: Force dropout for specific samples

        Returns:
            Embedded condition of shape [batch, dim_embedding]
        """
        month = y["month"]
        if month.dim() == 2:
            month = month.squeeze(-1)

        return self.month_embedder(month, train=train, force_drop_ids=force_drop_ids)

    @property
    def output_dim(self) -> int:
        return self.dim_out


@register_embedder("wpuq_month_season")
class WPuQMonthSeasonEmbedder(nn.Module):
    """Month and season embedder for WPuQ data.

    Embeds both month (0-11) and season (0-3) for richer conditioning.
    Season is derived from month: winter(0), spring(1), summer(2), autumn(3).

    Args:
        month: Dictionary with IntegerEmbedder arguments for month embedding.
        season: Dictionary with IntegerEmbedder arguments for season embedding.
    """

    def __init__(self, month: dict[str, Any], season: dict[str, Any] | None = None):
        super().__init__()

        if "num_embedding" not in month:
            month["num_embedding"] = 12

        self.month_embedder = IntegerEmbedder(**month)

        # Season embedder (optional)
        if season is not None:
            if "num_embedding" not in season:
                season["num_embedding"] = 4
            self.season_embedder = IntegerEmbedder(**season)
            self.use_season = True
            self.dim_out = month["dim_embedding"] + season["dim_embedding"]
        else:
            self.season_embedder = None
            self.use_season = False
            self.dim_out = month["dim_embedding"]

        # Month to season mapping
        # Winter: Dec, Jan, Feb (months 11, 0, 1) -> 0
        # Spring: Mar, Apr, May (months 2, 3, 4) -> 1
        # Summer: Jun, Jul, Aug (months 5, 6, 7) -> 2
        # Autumn: Sep, Oct, Nov (months 8, 9, 10) -> 3
        self.register_buffer(
            "month_to_season",
            torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]),  # 12 months
        )

    def forward(
        self,
        y: dict[str, torch.Tensor],
        train: bool | None = None,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed month and optionally season conditions.

        Args:
            y: Dictionary containing 'month' tensor
            train: Whether in training mode
            force_drop_ids: Force dropout for specific samples

        Returns:
            Embedded condition of shape [batch, total_dim]
        """
        month = y["month"]
        if month.dim() == 2:
            month = month.squeeze(-1)

        month_emb = self.month_embedder(month, train=train, force_drop_ids=force_drop_ids)

        if self.use_season:
            # Derive season from month
            season = self.month_to_season[month]
            season_emb = self.season_embedder(season, train=train, force_drop_ids=force_drop_ids)
            return torch.cat([month_emb, season_emb], dim=-1)

        return month_emb

    @property
    def output_dim(self) -> int:
        return self.dim_out


@register_embedder("wpuq_full")
class WPuQFullEmbedder(nn.Module):
    """Full embedder for WPuQ data with multiple condition types.

    Supports:
    - month: Month index (0-11)
    - season: Season index (0-3), can be derived from month
    - day_type: Weekday(0) vs Weekend(1)
    - household_id: Household identifier (for multi-household conditioning)

    Args:
        month: Dictionary with IntegerEmbedder arguments for month.
        season: Optional dictionary for season embedding.
        day_type: Optional dictionary for day type embedding.
        household_id: Optional dictionary for household ID embedding.
    """

    def __init__(
        self,
        month: dict[str, Any],
        season: dict[str, Any] | None = None,
        day_type: dict[str, Any] | None = None,
        household_id: dict[str, Any] | None = None,
    ):
        super().__init__()

        embedders = {}
        total_dim = 0

        # Month embedder (required)
        if "num_embedding" not in month:
            month["num_embedding"] = 12
        embedders["month"] = IntegerEmbedder(**month)
        total_dim += month["dim_embedding"]

        # Season embedder (optional)
        if season is not None:
            if "num_embedding" not in season:
                season["num_embedding"] = 4
            embedders["season"] = IntegerEmbedder(**season)
            total_dim += season["dim_embedding"]
            self.derive_season = True
            self.register_buffer(
                "month_to_season",
                torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]),
            )
        else:
            self.derive_season = False

        # Day type embedder (optional)
        if day_type is not None:
            if "num_embedding" not in day_type:
                day_type["num_embedding"] = 2  # weekday vs weekend
            embedders["day_type"] = IntegerEmbedder(**day_type)
            total_dim += day_type["dim_embedding"]

        # Household ID embedder (optional)
        if household_id is not None:
            embedders["household_id"] = IntegerEmbedder(**household_id)
            total_dim += household_id["dim_embedding"]

        self.embedders = nn.ModuleDict(embedders)
        self.dim_out = total_dim

    def forward(
        self,
        y: dict[str, torch.Tensor],
        train: bool | None = None,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed all available conditions.

        Args:
            y: Dictionary containing condition tensors
            train: Whether in training mode
            force_drop_ids: Force dropout for specific samples

        Returns:
            Concatenated embeddings of shape [batch, total_dim]
        """
        embeddings = []

        # Month (required)
        month = y["month"]
        if month.dim() == 2:
            month = month.squeeze(-1)
        embeddings.append(
            self.embedders["month"](month, train=train, force_drop_ids=force_drop_ids)
        )

        # Season (derived from month if needed)
        if "season" in self.embedders:
            if "season" in y:
                season = y["season"]
                if season.dim() == 2:
                    season = season.squeeze(-1)
            else:
                # Derive from month
                season = self.month_to_season[month]
            embeddings.append(
                self.embedders["season"](season, train=train, force_drop_ids=force_drop_ids)
            )

        # Day type (optional)
        if "day_type" in self.embedders and "day_type" in y:
            day_type = y["day_type"]
            if day_type.dim() == 2:
                day_type = day_type.squeeze(-1)
            embeddings.append(
                self.embedders["day_type"](day_type, train=train, force_drop_ids=force_drop_ids)
            )

        # Household ID (optional)
        if "household_id" in self.embedders and "household_id" in y:
            hh_id = y["household_id"]
            if hh_id.dim() == 2:
                hh_id = hh_id.squeeze(-1)
            embeddings.append(
                self.embedders["household_id"](hh_id, train=train, force_drop_ids=force_drop_ids)
            )

        return torch.cat(embeddings, dim=-1)

    @property
    def output_dim(self) -> int:
        return self.dim_out
