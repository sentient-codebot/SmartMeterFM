from ..configuration import ExperimentConfig
from ._mlflow import MLFlowModelLogger, create_mlflow_logger
from ._wandb import (
    WandbArtifactCleaner,
    WandbModelLogger,
    create_wandb_logger,
)
from .generic import GlobalProgressBar


__all__ = [
    "create_logger",
    "GlobalProgressBar",
    "WandbArtifactCleaner",
    "WandbModelLogger",
    "MLFlowModelLogger",
]


def create_logger(config: ExperimentConfig, run_id: str, logger_type: str = "wandb"):
    """Factory function to create the appropriate logger based on user preference.

    Args:
        config: Experiment configuration
        run_id: Unique identifier for the run
        logger_type: Type of logger to use ('wandb' or 'mlflow')

    Returns:
        A PyTorch Lightning logger instance
    """
    if logger_type.lower() == "wandb":
        return create_wandb_logger(config, run_id)
    elif logger_type.lower() == "mlflow":
        return create_mlflow_logger(config, run_id)
    else:
        raise ValueError(
            f"Unsupported logger_type: {logger_type}. Choose 'wandb' or 'mlflow'"
        )
