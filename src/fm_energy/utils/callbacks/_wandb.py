import os

import pytorch_lightning as pl
import torch
import wandb

from ..configuration import ExperimentConfig


class WandbArtifactCleanerAlternative(pl.Callback):
    """this deletes all artifact versions except the latest and best"""

    def __init__(
        self,
    ):
        super().__init__()

    def on_validation_end(self, trainer, pl_module):
        if not wandb.run:
            return

        api = wandb.Api()
        artifact_type = "model"
        artifact_name = f"{wandb.run.entity}/{wandb.run.project}/model-{wandb.run.id}"

        try:
            # Get collection of artifacts
            all_versions = list(api.artifacts(artifact_type, artifact_name))
            _latest = list(api.artifacts(artifact_type, artifact_name, tags="latest"))
            _best = list(api.artifacts(artifact_type, artifact_name, tags="best"))

            # Delete all versions except the latest and best
            versions_to_keep = set(_latest + _best)
            for version in all_versions:
                if version not in versions_to_keep:
                    version.delete()

        except Exception as e:
            print(f"Error cleaning artifacts: {e}")


class WandbArtifactCleaner(pl.Callback):
    def __init__(self, keep_n_latest=2):
        super().__init__()
        self.keep_n_latest = keep_n_latest

    def on_validation_end(self, trainer, pl_module):
        if not wandb.run:
            return

        api = wandb.Api()
        artifact_type = "model"
        artifact_name = f"{wandb.run.entity}/{wandb.run.project}/model-{wandb.run.id}"

        try:
            # Get all versions
            versions = list(api.artifacts(artifact_type, artifact_name))
            versions = sorted(versions, key=lambda x: x.created_at, reverse=True)

            # Find version marked as "best"
            best_version = next((v for v in versions if "best" in v.aliases), None)

            # Keep N latest versions and best version
            versions_to_keep = set(versions[: self.keep_n_latest])
            if best_version:
                versions_to_keep.add(best_version)

            # Delete versions not in keep set
            for version in versions:
                if version not in versions_to_keep:
                    version.delete()

        except Exception as e:
            print(f"Error cleaning artifacts: {e}")


class WandbModelLogger(pl.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not wandb.run:
            return

        # Create a temporary file to save the checkpoint
        checkpoint_path = os.path.join(trainer.default_root_dir, "temp_checkpoint.ckpt")
        torch.save(checkpoint, checkpoint_path)

        try:
            current_score = trainer.checkpoint_callback.current_score

            artifact = wandb.Artifact(
                name=f"{wandb.run.id}_model",
                type="model",
                metadata={"score": current_score},
            )

            artifact.add_file(checkpoint_path)

            aliases = ["latest"]
            if (
                trainer.checkpoint_callback.best_model_path
                == trainer.checkpoint_callback.last_model_path
            ):
                aliases.append("best")

            wandb.log_artifact(artifact, aliases=aliases)

        finally:
            # Clean up temporary file
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


def create_wandb_logger(
    config: ExperimentConfig, run_id: str
) -> pl.loggers.WandbLogger:
    r"""
    Setup and init the wandb logger.
    """
    wandb_logger = pl.loggers.WandbLogger(
        project={
            "wpuq": "HeatFM",
            "wpuq_trafo": "WPuQTrafoFM",
            "wpuq_pv": "WPuQPVFM",
            "lcl_electricity": "LCLFM",
            "cossmic": "CoSSMicFM",
        }[config.data.dataset],
        name=run_id,
        config=config.to_dict(),
        log_model="all",  # to be combined with WandbArtifactCleaner
    )
    return wandb_logger
