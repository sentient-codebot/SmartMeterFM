"""Masked Autoencoder for time series imputation."""

import math

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from einops import rearrange
from tqdm import tqdm

from ._base import NNImputeBaseline


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]


class MaskedAutoencoderModel(nn.Module):
    """Masked Autoencoder model for time series."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        folding_factor: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers
        self.folding_factor = folding_factor

        # Input projection (from folded dimensions to d_model)
        self.input_projection = nn.Linear(folding_factor, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        # Transformer decoder (only if num_decoder_layers > 0)
        assert num_decoder_layers > 0, "this version uses both enc-dec"
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers
        )

        # Output projection (from d_model back to folded dimensions)
        self.output_projection = nn.Linear(d_model, folding_factor)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of masked autoencoder.

        Args:
            x: Input time series [B, T, 1] (assuming single channel input)
            mask: Mask tensor [B, T, 1] where 1=observed, 0=missing

        Returns:
            Reconstructed time series [B, T, 1]
        """
        B, T, D = x.shape
        assert D == 1, f"Expected single channel input, got {D} channels"

        # Step 0: Normalize input
        mu = (x * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        )  # [B, 1, 1]
        std = ((x - mu) ** 2 * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        )  # [B, 1, 1]
        std = torch.sqrt(std + 1e-6)  # Avoid division by zero
        x = (x - mu) / std  # Normalize to zero mean, unit variance

        # Step 1: Fold the sequence dimension
        # [B, T, 1] -> [B, T//folding_factor, folding_factor]
        T_folded = T // self.folding_factor
        x_folded = rearrange(
            x,
            "B (T_fold fold) D -> B T_fold (fold D)",
            T_fold=T_folded,
            fold=self.folding_factor,
        )
        mask_folded = rearrange(
            mask,
            "B (T_fold fold) D -> B T_fold (fold D)",
            T_fold=T_folded,
            fold=self.folding_factor,
        )

        # Step 2: Project input to model dimension
        x_proj = self.input_projection(x_folded)  # [B, T_folded, d_model]

        # Create masked input (replace missing values with mask token)
        mask_expanded = mask_folded.any(
            dim=-1, keepdim=True
        )  # [B, T_folded, 1] - any feature observed

        masked_input = mask_expanded * x_proj + (~mask_expanded) * torch.zeros_like(
            x_proj
        )  # dummy tensor for missing values

        # Add positional encoding
        masked_input = self.pos_encoder(masked_input.transpose(0, 1)).transpose(0, 1)

        # Encoder
        enc_mask = ~mask_folded.any(dim=-1, keepdim=False)  # [B, T_folded]
        encoded = self.transformer_encoder(
            masked_input, src_key_padding_mask=enc_mask
        )  # [B, T_folded, d_model]

        # Decoder (if available) or use encoder output directly
        mask_tokens = self.mask_token.expand(B, T_folded, -1)  # [B, T_folded, d_model]
        input_with_mask_token = torch.where(
            mask_expanded,
            masked_input,
            mask_tokens,  # Use mask token for missing values
        )
        # positional encoding for decoder input
        input_with_mask_token = self.pos_encoder(
            input_with_mask_token.transpose(0, 1)
        ).transpose(0, 1)  # [B, T_folded, d_model]
        decoded = self.transformer_decoder(
            tgt=input_with_mask_token,
            memory=encoded,
            memory_key_padding_mask=enc_mask,  # only attend to observed positions
        )  # [B, T_folded, d_model]

        # Output projection
        reconstructed_folded = self.output_projection(
            decoded
        )  # [B, T_folded, folding_factor]

        # Step 3: Unfold back to original sequence length
        reconstructed = rearrange(
            reconstructed_folded,
            "B T_fold (fold D) -> B (T_fold fold) D",
            fold=self.folding_factor,
            D=1,
        )  # [B, T, 1]

        # Step 4: Rescale back to original scale
        reconstructed = reconstructed * std + mu  # Rescale to original mean, std

        return reconstructed


class MaskedAutoencoderBaseline(NNImputeBaseline):
    """Masked Autoencoder baseline for time series imputation."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        folding_factor: int = 1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        device: str | None = None,
        use_wandb: bool = False,
        wandb_project: str = "smartmeterfm-imputation",
        wandb_run_name: str | None = None,
    ):
        """
        Initialize Masked Autoencoder baseline.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            folding_factor: Factor to fold sequence dimension (reduces sequence length)
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            device: Device to use
            use_wandb: Whether to use wandb logging
            wandb_project: Wandb project name
            wandb_run_name: Wandb run name (optional)
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.folding_factor = folding_factor
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

    def _compute_loss(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor,
        only_unobserved: bool = True,
        rescale: bool = False,
    ):
        """
        Compute masked MSE loss.

        Args:
            reconstructed: Reconstructed time series [B, T, D]
            original: Original time series [B, T, D]
            mask: Mask tensor [B, T, D] where 1=observed, 0=missing
            only_unobserved: Whether to compute loss only on unobserved values
            rescale: Whether to rescale the data with mean/std before calculating loss
        """
        criterion = nn.MSELoss()
        if only_unobserved:
            reconstructed = reconstructed * (1 - mask)
            original = original * (1 - mask)
        if rescale:
            mu = (original * mask).sum(dim=1, keepdim=True) / mask.sum(
                dim=1, keepdim=True
            )
            _var = ((original - mu) ** 2 * mask).sum(dim=1, keepdim=True) / mask.sum(
                dim=1, keepdim=True
            )
            std = torch.sqrt(_var + 1e-6)  # Avoid division by zero
            reconstructed = (reconstructed - mu) / std
            original = (original - mu) / std
        loss = criterion(reconstructed, original)
        # ideally should divide by #missing values, but it's roughly constant.
        return loss

    def _initialize_wandb(self):
        """Initialize wandb if enabled."""
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_encoder_layers": self.num_encoder_layers,
                    "num_decoder_layers": self.num_decoder_layers,
                    "dim_feedforward": self.dim_feedforward,
                    "dropout": self.dropout,
                    "folding_factor": self.folding_factor,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "device": str(self.device),
                },
            )

    def _log_imputation_samples(self, batch_data, batch_masks, reconstructed, epoch):
        """Log sample imputation results to wandb."""
        if not self.use_wandb:
            return

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

    def fit(self, dataloader):
        """
        Fit Masked Autoencoder on training data using DataLoader.

        Args:
            dataloader: DataLoader that yields (ts_3d_batch, mask_3d_batch)
        """
        # Initialize wandb
        self._initialize_wandb()

        # Initialize model
        self.model = MaskedAutoencoderModel(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            folding_factor=self.folding_factor,
        ).to(self.device)

        # Print model parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            f"MaskedAutoencoder model initialized with {num_params:,} trainable parameters"
        )

        if self.use_wandb:
            wandb.log({"model_parameters": num_params})

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        pbar_epoch = tqdm(range(self.epochs), desc="MAE Training")
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
                reconstructed = self.model(batch_data, batch_masks)

                # Compute loss
                loss = self._compute_loss(
                    reconstructed,
                    batch_data,
                    batch_masks,
                    only_unobserved=False,
                    rescale=False,
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
                        sample_reconstructed = self.model(sample_data, sample_masks)

                    self._log_imputation_samples(
                        sample_data, sample_masks, sample_reconstructed, epoch + 1
                    )

            if (epoch + 1) % 20 == 0:
                print(f"MAE Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True

        # Close wandb run
        if self.use_wandb:
            wandb.finish()

    def fit_legacy(
        self,
        time_series_list: list[torch.Tensor],
        masks_list: list[torch.Tensor] | None = None,
    ):
        """Legacy fit method for backwards compatibility."""
        # Concatenate all training data
        all_data = torch.cat(time_series_list, dim=0)  # [N, T, D]

        if masks_list is not None:
            all_masks = torch.cat(masks_list, dim=0)  # [N, T, D]
            dataset = torch.utils.data.TensorDataset(all_data, all_masks)
        else:
            # Create random masks for self-supervised training
            random_masks = []
            for data_batch in time_series_list:
                batch_masks = self._create_random_mask(data_batch.shape, mask_ratio=0.3)
                random_masks.append(batch_masks)
            all_masks = torch.cat(random_masks, dim=0)
            dataset = torch.utils.data.TensorDataset(all_data, all_masks)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.fit(dataloader)

    def fit_dataloader(self, dataloader):
        """
        Fit Masked Autoencoder using a dataloader.

        Args:
            dataloader: DataLoader that yields (ts_3d_batch, mask_3d_batch)
        """
        print("Training Masked Autoencoder...")
        self.fit(dataloader)

    def impute(self, time_series: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Impute missing values using trained Masked Autoencoder.

        Args:
            time_series: Input time series [B, T, D]
            mask: Mask tensor [B, T, D] where 1=observed, 0=missing

        Returns:
            Imputed time series [B, T, D]
        """
        if not self.is_fitted:
            raise RuntimeError("Masked Autoencoder must be fitted before imputation")

        self.model.eval()
        with torch.no_grad():
            time_series = time_series.to(self.device)
            mask = mask.to(self.device)

            # Get reconstruction
            reconstructed = self.model(time_series, mask)

            # Combine observed values with reconstructed missing values
            imputed = mask * time_series + (1 - mask) * reconstructed

        return imputed.cpu()
