"""Baseline models for imputation and super-resolution tasks."""

from .imputation.brits import BRITSBaseline
from .imputation.knn_imputer import KNNImputerBaseline
from .imputation.load_pin import LoadPINBaseline
from .imputation.masked_autoencoder import MaskedAutoencoderBaseline
from .interpolation import InterpolationBaseline
from .super_resolution.gaussian_process import GaussianProcessBaseline
from .super_resolution.profile_sr import ProfileSRBaseline
from .super_resolution.temporal_cnn import TemporalCNNBaseline


__all__ = [
    "InterpolationBaseline",
    "KNNImputerBaseline",
    "LoadPINBaseline",
    "BRITSBaseline",
    "MaskedAutoencoderBaseline",
    "GaussianProcessBaseline",
    "TemporalCNNBaseline",
    "ProfileSRBaseline",
]
