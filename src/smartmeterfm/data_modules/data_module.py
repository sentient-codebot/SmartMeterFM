"""PyTorch Lightning DataModule for time series data."""

import os
from collections.abc import Callable, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .torch_dataset import Dataset1D


def get_optimal_num_workers(override=None):
    """
    Automatically determine optimal number of DataLoader workers based on environment.

    Args:
        override: If provided, overrides automatic detection

    Returns:
        int: Recommended number of workers
    """
    if override is not None:
        return override

    in_slurm = "SLURM_JOB_ID" in os.environ
    cpu_count = os.cpu_count() or 1
    num_gpus = torch.cuda.device_count()

    if in_slurm:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or int(
            os.environ.get("SLURM_CPUS_ON_NODE", 0)
        )
        if slurm_cpus > 0:
            return max(1, slurm_cpus - 1)
        else:
            return min(num_gpus * 4, max(1, int(cpu_count * 0.75)))
    else:
        return min(num_gpus * 2 or 2, max(1, int(cpu_count * 0.5)))


class TimeSeriesDataModule(pl.LightningDataModule):
    """DataModule that wraps a DatasetCollection into train/val/test DataLoaders.

    Args:
        data_collection: Object with a .dataset attribute (DatasetWithMetadata)
        batch_size: Batch size for all dataloaders
        labels: Label name(s) to extract from StaticLabelContainer
        profile_transform: Optional transform applied to profile tensors
        label_transform: Optional transform applied to label tensors
        collate_fn: Optional custom collate function for DataLoader
        num_workers: Override for number of DataLoader workers (None = auto)
    """

    def __init__(
        self,
        data_collection,
        batch_size: int = 32,
        labels: str | Sequence[str] | None = None,
        profile_transform: Callable | None = None,
        label_transform: Callable | None = None,
        collate_fn: Callable | None = None,
        num_workers: int | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_collection = data_collection
        self.labels = labels if labels is not None else []
        self.profile_transform = profile_transform
        self.label_transform = label_transform
        self.collate_fn = collate_fn
        self.num_workers = num_workers

    def setup(self, stage=""):
        for split_name in ["train", "val", "test"]:
            profile = self.dataset_collection.dataset.profile[split_name]
            label = self.dataset_collection.dataset.label[split_name][self.labels]
            if self.profile_transform is not None:
                profile = self.profile_transform(profile)
            if self.label_transform is not None:
                label = self.label_transform(label)
            setattr(self, f"{split_name}_dataset", Dataset1D(profile, label))

    def _make_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=get_optimal_num_workers(self.num_workers),
            pin_memory=True,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset, shuffle=False)

    def get_data_shape(self) -> tuple[int, ...]:
        """Return the shape of a single sample (seq_len, channels)."""
        return self.train_dataset.sample_shape
