import os

import pytorch_lightning as pl
import torch

from ..configuration import ExperimentConfig


class MLFlowModelLogger(pl.Callback):
    """Callback to log models to MLFlow during training"""

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not isinstance(trainer.logger, pl.loggers.MLFlowLogger):
            return

        if not trainer.is_global_zero:
            return

        # Create a temporary file to save the checkpoint
        checkpoint_path = os.path.join(
            trainer.default_root_dir, "model_checkpoint.ckpt"
        )
        torch.save(checkpoint, checkpoint_path)

        try:
            # Get the current validation score
            _current_score = trainer.checkpoint_callback.current_score

            # Log the model to MLFlow
            trainer.logger.experiment.log_artifact(
                trainer.logger.run_id, checkpoint_path, "model"
            )

            # Add tag if this is the best model
            if (
                trainer.checkpoint_callback.best_model_path
                == trainer.checkpoint_callback.last_model_path
            ):
                trainer.logger.experiment.set_tag(
                    trainer.logger.run_id, "best_model", "true"
                )

        finally:
            # Clean up temporary file
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


def create_mlflow_logger(
    config: ExperimentConfig, run_id: str
) -> pl.loggers.MLFlowLogger:
    """Setup and initialize the MLFlow logger.

    Args:
        config: Experiment configuration
        run_id: Unique identifier for the run

    Returns:
        PyTorch Lightning MLFlowLogger instance
    """
    # Map dataset names to experiment names
    experiment_name = {
        "wpuq": "HeatFM",
        "wpuq_trafo": "WPuQTrafoFM",
        "wpuq_pv": "WPuQPVFM",
        "lcl_electricity": "LCLFM",
        "cossmic": "CoSSMicFM",
    }[config.data.dataset]

    # Create the MLFlow logger
    mlflow_logger = pl.loggers.MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_id,
        tracking_uri=config.mlflow_tracking_uri
        if hasattr(config, "mlflow_tracking_uri")
        else None,
        tags={"time_id": config.time_id} if config.time_id else {},
    )

    # Log all configuration parameters
    mlflow_logger.log_hyperparams(config.to_dict())

    return mlflow_logger
