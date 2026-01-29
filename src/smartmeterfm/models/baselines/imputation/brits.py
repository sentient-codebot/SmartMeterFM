"""BRITS: Bidirectional Recurrent Imputation for Time Series."""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from ._base import NNImputeBaseline


class BRITSModel(nn.Module):
    """BRITS model implementation."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Forward LSTM
        self.lstm_forward = nn.LSTM(input_dim * 2, hidden_dim, batch_first=True)

        # Backward LSTM
        self.lstm_backward = nn.LSTM(input_dim * 2, hidden_dim, batch_first=True)

        # Output layers
        self.output_forward = nn.Linear(hidden_dim, input_dim)
        self.output_backward = nn.Linear(hidden_dim, input_dim)

        # Final combination layer
        self.combine = nn.Linear(input_dim * 2, input_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of BRITS.

        Args:
            x: Input time series [B, T, D]
            mask: Mask tensor [B, T, D] where 1=observed, 0=missing

        Returns:
            imputed: Final imputed time series [B, T, D]
            forward_imputed: Forward direction imputed [B, T, D]
            backward_imputed: Backward direction imputed [B, T, D]
        """
        batch_size, seq_len, input_dim = x.shape

        # Initialize imputed values with input
        forward_imputed = x.clone()
        backward_imputed = x.clone()

        # Forward direction
        forward_input = torch.cat([x, mask], dim=-1)  # [B, T, 2*D]
        forward_hidden, _ = self.lstm_forward(forward_input)  # [B, T, H]
        forward_output = self.output_forward(forward_hidden)  # [B, T, D]

        # Update forward imputed values
        forward_imputed = mask * x + (1 - mask) * forward_output

        # Backward direction (reverse time)
        backward_input = torch.cat([x.flip(dims=[1]), mask.flip(dims=[1])], dim=-1)
        backward_hidden, _ = self.lstm_backward(backward_input)
        backward_output = self.output_backward(backward_hidden).flip(
            dims=[1]
        )  # Flip back

        # Update backward imputed values
        backward_imputed = mask * x + (1 - mask) * backward_output

        # Combine forward and backward
        combined_input = torch.cat([forward_imputed, backward_imputed], dim=-1)
        final_imputed = self.combine(combined_input)

        # Apply mask to preserve observed values
        final_imputed = mask * x + (1 - mask) * final_imputed

        return final_imputed, forward_imputed, backward_imputed


class BRITSBaseline(NNImputeBaseline):
    """BRITS baseline for time series imputation."""

    def __init__(
        self,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        device: str | None = None,
        use_wandb: bool = False,
        wandb_project: str = "smartmeterfm-imputation",
        wandb_run_name: str | None = None,
    ):
        """
        Initialize BRITS baseline.

        Args:
            hidden_dim: Hidden dimension of LSTM layers
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to use ('cuda' or 'cpu')
            use_wandb: Whether to use wandb logging
            wandb_project: Wandb project name
            wandb_run_name: Wandb run name (optional)
        """
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.is_fitted = False

    def _compute_loss(
        self,
        imputed: torch.Tensor,
        forward_imputed: torch.Tensor,
        backward_imputed: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BRITS loss (reconstruction + consistency).

        Args:
            imputed: Final imputed values [B, T, D]
            forward_imputed: Forward direction imputed [B, T, D]
            backward_imputed: Backward direction imputed [B, T, D]
            target: True values [B, T, D]
            mask: Mask tensor [B, T, D]

        Returns:
            Total loss
        """
        # Reconstruction loss (only on observed values)
        recon_loss = torch.mean(mask * (imputed - target) ** 2)

        # Consistency loss (forward and backward should agree on missing values)
        missing_mask = 1 - mask
        consistency_loss = torch.mean(
            missing_mask * (forward_imputed - backward_imputed) ** 2
        )

        return recon_loss + 0.1 * consistency_loss

    def _initialize_wandb(self):
        """Initialize wandb if enabled."""
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "hidden_dim": self.hidden_dim,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "device": str(self.device),
                },
            )

    def _log_imputation_samples(
        self, batch_data, batch_masks, imputed, forward_imputed, backward_imputed, epoch
    ):
        """Log sample imputation results to wandb."""
        if not self.use_wandb:
            return

        # Take first sample from batch for visualization
        sample_original = batch_data[0].cpu().numpy()  # [T, D]
        sample_mask = batch_masks[0].cpu().numpy()  # [T, D]
        sample_imputed = imputed[0].cpu().numpy()  # [T, D]
        sample_forward = forward_imputed[0].cpu().numpy()  # [T, D]
        sample_backward = backward_imputed[0].cpu().numpy()  # [T, D]

        # Focus on first dimension for visualization
        if sample_original.shape[1] > 0:
            dim_idx = 0
            original_1d = sample_original[:, dim_idx]
            mask_1d = sample_mask[:, dim_idx]
            imputed_1d = sample_imputed[:, dim_idx]
            forward_1d = sample_forward[:, dim_idx]
            backward_1d = sample_backward[:, dim_idx]

            # Prepare data for plotting
            time_steps = list(range(len(original_1d)))

            # Create table data showing missing vs observed vs imputed
            table_data = []
            for t in range(len(original_1d)):
                is_observed = bool(mask_1d[t])
                table_data.append(
                    [
                        t,
                        float(original_1d[t]),
                        float(imputed_1d[t]),
                        "Observed" if is_observed else "Missing→Imputed",
                        float(forward_1d[t]),
                        float(backward_1d[t]),
                    ]
                )

            # Log as wandb table
            table = wandb.Table(
                data=table_data,
                columns=["Time", "Original", "Final", "Status", "Forward", "Backward"],
            )

            wandb.log({"train/imputation_sample": table, "train/epoch": epoch})

            # Also log a line plot showing the imputation
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(14, 4))

            # Plot original observed values
            observed_indices = mask_1d.astype(bool)
            missing_indices = ~observed_indices

            ax.plot(
                time_steps,
                original_1d,
                "o-",
                color="C2",
                label="Original (all)",
                alpha=0.3,
            )
            ax.scatter(
                [t for t, obs in enumerate(observed_indices) if obs],
                original_1d[observed_indices],
                color="blue",
                s=20,
                label="Observed values",
                zorder=5,
            )
            ax.scatter(
                [t for t, miss in enumerate(missing_indices) if miss],
                imputed_1d[missing_indices],
                marker="^",
                color="red",
                s=20,
                label="Imputed values",
                zorder=5,
            )
            ax.scatter(
                [t for t, miss in enumerate(missing_indices) if miss],
                forward_1d[missing_indices],
                marker="^",
                color="green",
                alpha=0.7,
                s=20,
                label="Forward direction",
                zorder=4,
            )
            ax.scatter(
                [t for t, miss in enumerate(missing_indices) if miss],
                backward_1d[missing_indices],
                marker="v",
                color="orange",
                alpha=0.7,
                s=20,
                label="Backward direction",
                zorder=4,
            )

            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.set_title(f"BRITS Imputation Example - Epoch {epoch}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            wandb.log({"train/imputation_plot": wandb.Image(fig)})
            plt.close(fig)

    def fit(self, dataloader):
        """
        Fit BRITS model on training data using DataLoader.

        Args:
            dataloader: DataLoader that yields (ts_3d_batch, mask_3d_batch)
        """
        # Initialize wandb
        self._initialize_wandb()
        # Determine input dimension from first batch
        first_batch = next(iter(dataloader))
        ts_batch, _ = first_batch
        input_dim = ts_batch.shape[-1]

        # Initialize model
        self.model = BRITSModel(input_dim, self.hidden_dim).to(self.device)

        # Print model parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"BRITS model initialized with {num_params:,} trainable parameters")

        if self.use_wandb:
            wandb.log({"model_parameters": num_params})

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        pbar_epoch = tqdm(range(self.epochs), desc="BRITS Training")
        for epoch in pbar_epoch:
            total_loss = 0
            num_batches = 0

            pbar_batch = tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False
            )
            for batch_data, batch_masks in pbar_batch:
                batch_data = batch_data.to(self.device)
                batch_masks = batch_masks.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                imputed, forward_imputed, backward_imputed = self.model(
                    batch_data, batch_masks
                )

                # Compute loss
                loss = self._compute_loss(
                    imputed, forward_imputed, backward_imputed, batch_data, batch_masks
                )

                # Backward pass
                loss.backward()
                optimizer.step()

                pbar_batch.set_postfix(loss=loss.item())
                if self.use_wandb:
                    wandb.log({"train/batch_loss": loss.item()})
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            pbar_epoch.set_postfix(loss=avg_loss)

            # Log to wandb
            if self.use_wandb:
                wandb.log({"train/epoch_loss": avg_loss, "train/epoch": epoch + 1})

                # Log imputation samples every 1 epochs or on the last epoch
                if (epoch + 1) % 1 == 0 or epoch == self.epochs - 1:
                    # Get a batch for sample logging
                    sample_batch = next(iter(dataloader))
                    sample_data, sample_masks = sample_batch
                    sample_data = sample_data.to(self.device)
                    sample_masks = sample_masks.to(self.device)

                    with torch.no_grad():
                        sample_imputed, sample_forward, sample_backward = self.model(
                            sample_data, sample_masks
                        )

                    self._log_imputation_samples(
                        sample_data,
                        sample_masks,
                        sample_imputed,
                        sample_forward,
                        sample_backward,
                        epoch + 1,
                    )

            if (epoch + 1) % 20 == 0:
                print(f"BRITS Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True

        # Close wandb run
        if self.use_wandb:
            wandb.finish()

    def fit_legacy(
        self, time_series_list: list[torch.Tensor], masks_list: list[torch.Tensor]
    ):
        """Legacy fit method for backwards compatibility."""
        # Concatenate all training data
        all_data = torch.cat(time_series_list, dim=0)  # [N, T, D]
        all_masks = torch.cat(masks_list, dim=0)  # [N, T, D]

        # Create DataLoader and use new fit method
        dataset = torch.utils.data.TensorDataset(all_data, all_masks)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.fit(dataloader)

    def fit_dataloader(self, dataloader):
        """
        Fit BRITS model using a dataloader.

        Args:
            dataloader: DataLoader that yields (ts_3d_batch, mask_3d_batch)
        """
        print("Training BRITS model...")
        self.fit(dataloader)

    def impute(self, time_series: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Impute missing values using trained BRITS model.

        Args:
            time_series: Input time series [B, T, D]
            mask: Mask tensor [B, T, D] where 1=observed, 0=missing

        Returns:
            Imputed time series [B, T, D]
        """
        if not self.is_fitted:
            raise RuntimeError("BRITS model must be fitted before imputation")

        self.model.eval()
        with torch.no_grad():
            time_series = time_series.to(self.device)
            mask = mask.to(self.device)

            imputed, _, _ = self.model(time_series, mask)

        return imputed.cpu()
