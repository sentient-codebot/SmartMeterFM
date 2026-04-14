"""Backward-compatible re-exports.

All symbols that were previously defined in this file are now split across
focused modules. This file re-exports them so existing imports continue to work.
"""

# Containers
# Base classes
from .base import (
    RESOLUTION_SECONDS,
    DatasetCollection,
    TimeSeriesDataCollection,
)
from .containers import (
    DatasetWithMetadata,
    DataShapeType,
    StaticLabelContainer,
)
from .data_module import (
    TimeSeriesDataModule,
    get_optimal_num_workers,
)

# Preprocessing / seasons
from .preprocessing import (
    NAME_SEASONS,
    months_of_season,
    season_of_month,
    shuffle_array,
    split_and_save_npz,
)

# Readers
from .readers import (
    BaseHDF5Reader,
    WPuQHouseholdReader,
    WPuQPVReader,
    WPuQReader,
)

# Statistics
from .statistics import (
    PIT,
    LogNorm,
    MultiDimECDF,
    MultiDimECDFV1,
    MultiDimECDFV2,
    standard_normal_cdf,
    standard_normal_icdf,
)

# Dataset / DataModule
from .torch_dataset import Dataset1D

# Transforms
from .transforms import (
    ChronoVectorize,
    Compose,
    MeanStdScaler,
    MinMaxScaler,
    Patchify,
    Transform,
)


__all__ = [
    # Containers
    "StaticLabelContainer",
    "DatasetWithMetadata",
    "DataShapeType",
    # Preprocessing
    "NAME_SEASONS",
    "months_of_season",
    "season_of_month",
    "shuffle_array",
    "split_and_save_npz",
    # Base
    "DatasetCollection",
    "TimeSeriesDataCollection",
    "RESOLUTION_SECONDS",
    # Transforms
    "Transform",
    "Compose",
    "MinMaxScaler",
    "MeanStdScaler",
    "Patchify",
    "ChronoVectorize",
    # Statistics
    "MultiDimECDFV2",
    "MultiDimECDFV1",
    "MultiDimECDF",
    "LogNorm",
    "PIT",
    "standard_normal_cdf",
    "standard_normal_icdf",
    # Readers
    "BaseHDF5Reader",
    "WPuQReader",
    "WPuQHouseholdReader",
    "WPuQPVReader",
    # Dataset / DataModule
    "Dataset1D",
    "TimeSeriesDataModule",
    "get_optimal_num_workers",
]
