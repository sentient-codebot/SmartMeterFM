"""MAE-LoadPIN hybrid model that uses a pretrained Masked Autoencoder as the coarse network
and the LoadPIN refine network for final imputation.
"""

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from tqdm import tqdm

from ._base import NNImputeBaseline
from .load_pin import LoadPINDiscriminator, LoadPINRefine
from .masked_autoencoder import MaskedAutoencoderModel


class MAELoadPINBaseline(NNImputeBaseline):
    """MAE-LoadPIN baseline model that uses pretrained MAE as coarse network."""

    def __init__(
        self,
        sequence_length: int,
        refine_loss_coef: tuple[float, float],
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        device: str | None = None,
        use_wandb: bool = False,
        wandb_project: str = "fm-energy-imputation",
        wandb_run_name: str | None = None,
        # MAE parameters
        mae_d_model: int = 128,
        mae_nhead: int = 8,
        mae_num_encoder_layers: int = 6,
        mae_num_decoder_layers: int = 6,
        mae_dim_feedforward: int = 512,
        mae_dropout: float = 0.1,
        mae_folding_factor: int = 1,
    ):
        """
        Args:
            sequence_length (int): Length of the input sequence.
            refine_loss_coef (tuple[float, float]): lambda_1, lambda_2 ~ adv, feat
            learning_rate (float): Learning rate for optimizers.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            device (str | None): Device to use for training.
            use_wandb (bool): Whether to use wandb logging.
            wandb_project (str): Wandb project name.
            wandb_run_name (str | None): Wandb run name (optional).
            mae_* parameters: Parameters for the pretrained MAE model.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.refine_loss_coef = refine_loss_coef
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        # MAE parameters
        self.mae_d_model = mae_d_model
        self.mae_nhead = mae_nhead
        self.mae_num_encoder_layers = mae_num_encoder_layers
        self.mae_num_decoder_layers = mae_num_decoder_layers
        self.mae_dim_feedforward = mae_dim_feedforward
        self.mae_dropout = mae_dropout
        self.mae_folding_factor = mae_folding_factor

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize networks
        self.mae_coarse = None  # Will be loaded from pretrained model
        self.refine = LoadPINRefine().to(self.device)
        self.discriminator = LoadPINDiscriminator(sequence_length).to(self.device)

        # Initialize optimizers (only for refine and discriminator, MAE is frozen)
        self.optimizer_refine = torch.optim.Adam(
            self.refine.parameters(), lr=self.learning_rate
        )
        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )

        self.is_fitted = False

    def load_pretrained_mae(self, mae_model_path: str) -> None:
        """
        Load a pretrained MAE model and freeze it as the coarse network.

        Args:
            mae_model_path (str): Path to the pretrained MAE model (.pt file containing MaskedAutoencoderBaseline)
        """
        print(f"Loading pretrained MAE from {mae_model_path}")

        # Initialize MAE model
        self.mae_coarse = MaskedAutoencoderModel(
            d_model=self.mae_d_model,
            nhead=self.mae_nhead,
            num_encoder_layers=self.mae_num_encoder_layers,
            num_decoder_layers=self.mae_num_decoder_layers,
            dim_feedforward=self.mae_dim_feedforward,
            dropout=self.mae_dropout,
            folding_factor=self.mae_folding_factor,
        ).to(self.device)

        # Load pretrained weights
        checkpoint = torch.load(
            mae_model_path, map_location=self.device, weights_only=False
        )

        if hasattr(checkpoint, "model") and checkpoint.model is not None:
            # Extract the MaskedAutoencoderModel from MaskedAutoencoderBaseline
            self.mae_coarse = checkpoint.model.to(self.device)
            print("Loaded MAE model from MaskedAutoencoderBaseline.model")
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.mae_coarse.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded MAE model from checkpoint state dict")
        else:
            # Assume the checkpoint is the state dict directly or unsupported format
            try:
                self.mae_coarse.load_state_dict(checkpoint)
                print("Loaded MAE model from direct state dict")
            except Exception as e:
                raise ValueError(
                    "Unsupported checkpoint format. Expected MaskedAutoencoderBaseline object, state dict, or dict with 'model_state_dict'. Error"
                ) from e

        # Freeze MAE parameters
        for param in self.mae_coarse.parameters():
            param.requires_grad = False

        # Set to evaluation mode
        self.mae_coarse.eval()

        mae_params = sum(p.numel() for p in self.mae_coarse.parameters())
        print(f"Loaded and frozen MAE with {mae_params:,} parameters")

    @staticmethod
    def _enable_grad(module: nn.Module, requires_grad: bool = True) -> None:
        """
        Enable/Disable gradient computation for the module.
        Args:
            module: nn.Module instance.
            requires_grad: If True, enables gradient computation; otherwise, disables it.
        """
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _compute_normalization_stats(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mu and std for normalization, similar to LoadPIN coarse network.

        Args:
            x: Input tensor [B, T, 1]
            mask: Mask tensor [B, T, 1] where 1=observed

        Returns:
            mu, std: Normalization statistics [B, 1, 1]
        """
        mu = (x * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        )  # [B, 1, 1]
        std = ((x - mu) ** 2 * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        )  # [B, 1, 1]
        std = torch.sqrt(std + 1e-6)  # Avoid division by zero
        return mu, std

    def _loss_refine_recon(
        self,
        x_gen: torch.Tensor,
        x_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine reconstruction loss function.
        Args:
            x_gen: Generated tensor by LoadPINRefine. shape (batch_size, seq_len, 1)
            x_real: Real tensor of shape (batch_size, seq_len, 1)
        Returns:
            RMSE value.
        """
        return torch.sqrt(F.mse_loss(x_gen, x_real))

    def _loss_refine_adv(
        self,
        realness: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine adversarial loss function.
        Args:
            realness: Realness tensor from discriminator. shape (batch_size, 1)
        Returns:
            Adversarial loss value.
        """
        return -realness.mean()

    def _loss_feat(
        self,
        features_gen: list[torch.Tensor],
        features_real: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Feature matching loss function.
        Args:
            features_gen: List of features from the discriminator for generated samples.
            features_real: List of features from the discriminator for real samples.
        Returns:
            Feature matching loss value.
        """
        assert len(features_gen) == len(
            features_real
        ), "Feature lists must have the same length."
        loss = 0.0
        for f_gen, f_real in zip(features_gen, features_real, strict=True):
            loss += torch.sqrt(F.mse_loss(f_gen, f_real)).mean()
        return loss / len(features_gen)

    def _loss_refine(
        self,
        x_gen: torch.Tensor,
        x_real: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Refine loss function.
        Args:
            x_gen: Generated tensor by LoadPINRefine. shape (batch_size, seq_len, 1)
            x_real: Real tensor of shape (batch_size, seq_len, 1)
        Returns:
            - combined loss value.
            - recon loss
            - adversarial loss
            - feature matching loss
        """
        # Reconstruction loss
        recon_loss = self._loss_refine_recon(x_gen, x_real)

        self._enable_grad(self.discriminator, requires_grad=False)
        # Adversarial loss
        realness, features_gen = self.discriminator(x_gen)
        adv_loss = self._loss_refine_adv(realness)

        # Feature matching loss
        _real_realness, features_real = self.discriminator(x_real)
        feat_loss = self._loss_feat(features_gen, features_real)

        return (
            (
                recon_loss
                + self.refine_loss_coef[0] * adv_loss
                + self.refine_loss_coef[1] * feat_loss
            ),
            recon_loss,
            adv_loss,
            feat_loss,
        )

    def _loss_discriminator(
        self,
        x_gen: torch.Tensor,
        x_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminator loss function.
        Args:
            x_gen: Generated tensor by LoadPINRefine. shape (batch_size, seq_len, 1)
            x_real: Real tensor of shape (batch_size, seq_len, 1)
        Returns:
            Total loss value.
        """
        # Realness for real samples
        realness_real, _ = self.discriminator(x_real)

        # Realness for generated samples
        realness_gen, _ = self.discriminator(x_gen)

        return (F.relu(1 - realness_real) + F.relu(1 + realness_gen)).mean()

    def _initialize_wandb(self):
        """Initialize wandb if enabled."""
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "sequence_length": self.sequence_length,
                    "refine_loss_coef": self.refine_loss_coef,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "device": str(self.device),
                    "mae_d_model": self.mae_d_model,
                    "mae_nhead": self.mae_nhead,
                    "mae_num_encoder_layers": self.mae_num_encoder_layers,
                    "mae_num_decoder_layers": self.mae_num_decoder_layers,
                },
            )

    def _log_imputation_samples(
        self, batch_data, batch_masks, x_coarse, x_refined, epoch
    ):
        """Log sample imputation results to wandb."""
        if not self.use_wandb:
            return

        # Take first sample from batch for visualization
        sample_original = batch_data[0].cpu().numpy()  # [T, 1]
        sample_mask = batch_masks[0].cpu().numpy()  # [T, 1]
        sample_coarse = x_coarse[0].cpu().numpy()  # [T, 1]
        sample_refined = x_refined[0].cpu().numpy()  # [T, 1]

        # Create imputed version (refined output for missing values)
        sample_imputed = (
            sample_mask * sample_original + (1 - sample_mask) * sample_refined
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
                    float(sample_coarse[t, 0]),
                    float(sample_refined[t, 0]),
                ]
            )

        # Log as wandb table
        table = wandb.Table(
            data=table_data,
            columns=["Time", "Original", "Final", "Status", "MAE_Coarse", "Refined"],
        )

        wandb.log({"train/imputation_sample": table, "train/epoch": epoch})

        # Also log a line plot showing the imputation
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Plot original observed values
        observed_indices = sample_mask[:, 0].astype(bool)
        missing_indices = ~observed_indices

        # Top plot: Original vs Imputed
        ax1.plot(
            time_steps,
            sample_original[:, 0],
            "o-",
            color="C2",
            label="Original (all)",
            alpha=0.3,
        )
        ax1.scatter(
            [t for t, obs in enumerate(observed_indices) if obs],
            sample_original[observed_indices, 0],
            color="blue",
            s=20,
            label="Observed values",
            zorder=5,
        )
        ax1.scatter(
            [t for t, miss in enumerate(missing_indices) if miss],
            sample_imputed[missing_indices, 0],
            marker="^",
            color="red",
            s=20,
            label="Imputed values (refined)",
            zorder=5,
        )
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Value")
        ax1.set_title(f"MAE-LoadPIN Imputation - Final Result - Epoch {epoch}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: MAE Coarse vs Refined generation
        ax2.plot(
            time_steps,
            sample_coarse[:, 0],
            "o-",
            color="orange",
            label="MAE Coarse output",
            alpha=0.7,
        )
        ax2.plot(
            time_steps,
            sample_refined[:, 0],
            "s-",
            color="purple",
            label="Refined output",
            alpha=0.7,
        )
        ax2.scatter(
            [t for t, obs in enumerate(observed_indices) if obs],
            sample_original[observed_indices, 0],
            color="blue",
            s=15,
            label="Observed values",
            zorder=5,
        )
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Value")
        ax2.set_title("Generator Outputs Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        wandb.log({"train/imputation_plot": wandb.Image(fig)})
        plt.close(fig)

    def _train_refine_step(
        self,
        x_real: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Train refine generator for one step.
        Args:
            x_real: Real tensor of shape (batch_size, seq_len, 1)
            mask: Mask tensor of shape (batch_size, seq_len, 1). 1=observed.
        Returns:
            Tuple of (coarse_output, refine_loss, refine_recon_loss, refine_adv_loss, refine_feat_loss).
        """
        self.optimizer_refine.zero_grad()

        # Generate coarse output using frozen MAE (no gradients)
        with torch.no_grad():
            x_coarse = self.mae_coarse(x_real, mask)
            # Compute normalization stats manually for refine network
            mu, std = self._compute_normalization_stats(x_real, mask)

        # Generate refined output
        x_refined = self.refine(x_coarse, mu, std)

        # Compute refine loss (includes adversarial and feature matching)
        refine_loss, refine_recon_loss, refine_adv_loss, refine_feat_loss = (
            self._loss_refine(x_refined, x_real)
        )

        # Backward and optimize
        refine_loss.backward()
        self.optimizer_refine.step()

        return (
            x_coarse,
            refine_loss,
            refine_recon_loss,
            refine_adv_loss,
            refine_feat_loss,
        )

    def _train_discriminator_step(
        self,
        x_real: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Train discriminator for one step.
        Args:
            x_real: Real tensor of shape (batch_size, seq_len, 1)
            mask: Mask tensor of shape (batch_size, seq_len, 1). 1=observed.
        Returns:
            Discriminator loss.
        """
        self.optimizer_discriminator.zero_grad()

        # Enable discriminator gradients
        self._enable_grad(self.discriminator, requires_grad=True)

        # Generate fake samples (detached to prevent generator gradients)
        with torch.no_grad():
            x_coarse = self.mae_coarse(x_real, mask)
            mu, std = self._compute_normalization_stats(x_real, mask)
            x_gen = self.refine(x_coarse, mu, std)

        # Compute discriminator loss
        disc_loss = self._loss_discriminator(x_gen, x_real)

        # Backward and optimize
        disc_loss.backward()
        self.optimizer_discriminator.step()

        return disc_loss

    def fit(self, dataloader):
        """
        Fit MAE-LoadPIN model on training data using DataLoader.

        Args:
            dataloader: DataLoader that yields (ts_3d_batch, mask_3d_batch)
        """
        if self.mae_coarse is None:
            raise RuntimeError(
                "Must load pretrained MAE first using load_pretrained_mae()"
            )

        # Initialize wandb
        self._initialize_wandb()

        print("Training MAE-LoadPIN model...")

        # Print model parameters
        mae_params = sum(p.numel() for p in self.mae_coarse.parameters())
        refine_params = sum(
            p.numel() for p in self.refine.parameters() if p.requires_grad
        )
        disc_params = sum(
            p.numel() for p in self.discriminator.parameters() if p.requires_grad
        )
        trainable_params = refine_params + disc_params

        print("MAE-LoadPIN model initialized:")
        print(f"  - MAE (frozen): {mae_params:,}")
        print(f"  - Refine: {refine_params:,}")
        print(f"  - Discriminator: {disc_params:,}")
        print(f"  - Total trainable: {trainable_params:,}")

        if self.use_wandb:
            wandb.log(
                {
                    "model_parameters_mae_frozen": mae_params,
                    "model_parameters_refine": refine_params,
                    "model_parameters_discriminator": disc_params,
                    "model_parameters_trainable": trainable_params,
                }
            )

        # Training loop
        pbar_epoch = tqdm(range(self.epochs), desc="MAE-LoadPIN Training")
        for epoch in pbar_epoch:
            total_refine_loss = 0
            total_disc_loss = 0
            num_batches = 0

            pbar_batch = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False
            )
            for batch_data, batch_masks in pbar_batch:
                batch_data = batch_data.to(self.device)
                batch_masks = batch_masks.to(self.device)

                # Train refine generator
                (
                    x_coarse,
                    refine_loss,
                    refine_recon_loss,
                    refine_adv_loss,
                    refine_feat_loss,
                ) = self._train_refine_step(batch_data, batch_masks)
                total_refine_loss += refine_loss.item()

                # Train discriminator
                disc_loss = self._train_discriminator_step(batch_data, batch_masks)
                total_disc_loss += disc_loss.item()

                # Log batch losses to wandb
                if self.use_wandb:
                    wandb.log(
                        {
                            "train/batch_refine_loss": refine_loss.item(),
                            "train/batch_refine_recon_loss": refine_recon_loss.item(),
                            "train/batch_refine_adv_loss": refine_adv_loss.item(),
                            "train/batch_refine_feat_loss": refine_feat_loss.item(),
                            "train/batch_disc_loss": disc_loss.item(),
                        }
                    )

                # Update progress bar with current losses
                pbar_batch.set_postfix(
                    {
                        "R": f"{refine_loss.item():.3f}",
                        "D": f"{disc_loss.item():.3f}",
                    }
                )

                num_batches += 1

            # Average losses for the epoch
            avg_refine_loss = total_refine_loss / num_batches
            avg_disc_loss = total_disc_loss / num_batches

            # Update epoch progress bar
            pbar_epoch.set_postfix(
                {
                    "RefineL": f"{avg_refine_loss:.3f}",
                    "DiscL": f"{avg_disc_loss:.3f}",
                }
            )

            # Log to wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "train/epoch_refine_loss": avg_refine_loss,
                        "train/epoch_disc_loss": avg_disc_loss,
                        "train/epoch": epoch + 1,
                    }
                )

                # Log imputation samples every epoch or on the last epoch
                if (epoch + 1) % 1 == 0 or epoch == self.epochs - 1:
                    # Get a batch for sample logging
                    sample_batch = next(iter(dataloader))
                    sample_data, sample_masks = sample_batch
                    sample_data = sample_data.to(self.device)
                    sample_masks = sample_masks.to(self.device)

                    with torch.no_grad():
                        sample_coarse = self.mae_coarse(sample_data, sample_masks)
                        sample_mu, sample_std = self._compute_normalization_stats(
                            sample_data, sample_masks
                        )
                        sample_refined = self.refine(
                            sample_coarse, sample_mu, sample_std
                        )

                    self._log_imputation_samples(
                        sample_data,
                        sample_masks,
                        sample_coarse,
                        sample_refined,
                        epoch + 1,
                    )

            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"MAE-LoadPIN Epoch {epoch + 1}/{self.epochs}")
                print(f"  Refine Loss: {avg_refine_loss:.4f}")
                print(f"  Disc Loss: {avg_disc_loss:.4f}")

        self.is_fitted = True
        print("MAE-LoadPIN training completed.")

        # Close wandb run
        if self.use_wandb:
            wandb.finish()

    def impute(
        self, time_series: torch.Tensor, mask: torch.Tensor, stochastic: bool = False
    ) -> torch.Tensor:
        """
        Impute missing values using trained MAE-LoadPIN model.

        Args:
            time_series: Input time series [B, T, D]
            mask: Mask tensor [B, T, D] where 1=observed, 0=missing
            stochastic: If True, adds noise for diverse samples; if False, deterministic
                       (Note: stochastic not implemented for MAE coarse network)

        Returns:
            Imputed time series [B, T, D]
        """
        if not self.is_fitted:
            raise RuntimeError("MAE-LoadPIN model must be fitted before imputation")

        if self.mae_coarse is None:
            raise RuntimeError(
                "Must load pretrained MAE first using load_pretrained_mae()"
            )

        # Set models to evaluation mode
        self.mae_coarse.eval()
        self.refine.eval()

        with torch.no_grad():
            time_series = time_series.to(self.device)
            mask = mask.to(self.device)

            # Generate coarse output using MAE
            x_coarse = self.mae_coarse(time_series, mask)

            # Compute normalization stats
            mu, std = self._compute_normalization_stats(time_series, mask)

            # Generate refined output
            x_refined = self.refine(x_coarse, mu, std)

            # Combine observed values with generated missing values
            imputed = mask * time_series + (1 - mask) * x_refined

        return imputed.cpu()
