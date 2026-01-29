from abc import ABC, abstractmethod

import torch


class NNImputeBaseline(ABC):
    """
    Abstract base class for neural network-based imputation baselines.
    """

    def _create_random_mask(
        self, shape: tuple, mask_ratio: float = 0.3
    ) -> torch.Tensor:
        """
        Create random mask for training.

        Args:
            shape: Shape of the mask [B, T, D]
            mask_ratio: Ratio of values to mask

        Returns:
            Random mask tensor
        """
        B, T, D = shape
        mask = torch.ones(B, T, D)

        # Randomly mask some positions
        for b in range(B):
            num_mask = int(T * mask_ratio)
            mask_indices = torch.randperm(T)[:num_mask]
            mask[b, mask_indices, :] = 0

        return mask

    def _log_imputation_samples(self, batch_data, batch_masks, reconstructed, epoch):
        """Log sample imputation results to wandb."""
        if not self.use_wandb:
            return

        import wandb

        # Take first sample from batch for visualization
        sample_original = batch_data[0].cpu().numpy()  # [T, 1]
        sample_mask = batch_masks[0].cpu().numpy()  # [T, 1]
        sample_reconstructed = reconstructed[0].cpu().numpy()  # [T, 1]

        # Create imputed version
        sample_imputed = (
            sample_mask * sample_original + (1 - sample_mask) * sample_reconstructed
        )

        # Prepare data for plotting
        time_steps = list(range(len(sample_original)))

        # Create table data showing missing vs observed vs imputed
        table_data = []
        for t in range(len(sample_original)):
            is_observed = bool(sample_mask[t, 0])
            table_data.append(
                [
                    t,
                    float(sample_original[t, 0]),
                    float(sample_imputed[t, 0]),
                    "Observed" if is_observed else "Missing→Imputed",
                    float(sample_reconstructed[t, 0]),
                ]
            )

        # Log as wandb table
        table = wandb.Table(
            data=table_data,
            columns=["Time", "Original", "Final", "Status", "Reconstructed"],
        )

        wandb.log({"train/imputation_sample": table, "train/epoch": epoch})

        # Also log a line plot showing the imputation
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        # Plot original observed values
        observed_indices = sample_mask[:, 0].astype(bool)
        missing_indices = ~observed_indices

        ax.plot(
            time_steps,
            sample_original[:, 0],
            "o-",
            color="C2",
            label="Original (all)",
            alpha=0.3,
        )
        ax.scatter(
            [t for t, obs in enumerate(observed_indices) if obs],
            sample_original[observed_indices, 0],
            color="blue",
            s=20,
            label="Observed values",
            zorder=5,
        )
        ax.scatter(
            [t for t, miss in enumerate(missing_indices) if miss],
            sample_imputed[missing_indices, 0],
            marker="^",
            color="red",
            s=20,
            label="Imputed values",
            zorder=5,
        )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.set_title(f"Imputation Example - Epoch {epoch}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        wandb.log({"train/imputation_plot": wandb.Image(fig)})
        plt.close(fig)

    @abstractmethod
    def fit(self, dataloader):
        """
        Fit the NN model on training data using DataLoader.

        Args:
            dataloader: DataLoader that yields (ts_3d_batch, mask_3d_batch)
        """
        raise NotImplementedError
        ...

    @abstractmethod
    def impute(
        self, time_series: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Impute missing values using trained model.

        Args:
            time_series: Input time series [B, T, D]
            mask: Mask tensor [B, T, D] where 1=observed, 0=missing
            **kwargs: Additional model-specific parameters

        Returns:
            Imputed time series [B, T, D]
        """
        raise NotImplementedError
        ...
