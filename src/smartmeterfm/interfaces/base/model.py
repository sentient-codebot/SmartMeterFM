from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel
from torch import nn

from .config import ModelConfig, SampleConfig


class ModelInterface(ABC):
    """
    Base interface for all model implementations.
    """

    @abstractmethod
    def __init__(self, model_config: ModelConfig):
        """Initialize the model with the given configuration."""
        pass

    @abstractmethod
    def _load_model(self) -> nn.Module:
        """Load model from storage."""
        pass

    @abstractmethod
    def sample(
        self,
        sample_config: SampleConfig,
        use_conditional: bool = False,
        conditional_config: BaseModel | None = None,
        condition: BaseModel | None = None,
        use_posterior: bool = False,
        posterior_config: BaseModel | None = None,
        measurement_y: torch.Tensor | None = None,
        x_0: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from loaded model with the given configuration."""
        pass
