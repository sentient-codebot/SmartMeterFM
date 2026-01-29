"""This is a reproduction of the Load-PIN model proposed in
Yiyan et al. Load Profile Inpainting for Missing Load Data Restoration and Baseline
Estimation. IEEE Trans. Smart Grid, 2024.
"""

from collections import namedtuple

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.nn.utils import spectral_norm
from tqdm import tqdm

from ._base import NNImputeBaseline


ConvLayerConfig = namedtuple(
    "ConvLayerConfig",
    [
        "transpose",
        "kernel_size",
        "in_channels",
        "out_channels",
        "stride",
    ],
)
AttentionLayerConfig = namedtuple(
    "AttentionLayerConfig",
    [
        "d_model",
        "nhead",
    ],
)


__LOAD_PIN_CONFIG__ = {
    "coarse": [
        ConvLayerConfig(False, 3, 1, 64, 1),  # [B, 64, T]
        ConvLayerConfig(False, 3, 64, 128, 2),  # [B, 128, T/2] - downsample
        ConvLayerConfig(False, 3, 128, 128, 1),  # [B, 128, T/2]
        ConvLayerConfig(False, 3, 128, 256, 1),  # [B, 256, T/2] - no upsample
        ConvLayerConfig(False, 3, 256, 256, 1),  # [B, 256, T/2]
        ConvLayerConfig(True, 3, 256, 128, 2),  # [B, 128, T] - upsample back
        ConvLayerConfig(False, 3, 128, 64, 1),  # [B, 64, T]
        ConvLayerConfig(False, 3, 64, 64, 1),  # [B, 64, T]
        ConvLayerConfig(False, 3, 64, 1, 1),  # [B, 1, T] - final output
    ],
    "refine": [
        ConvLayerConfig(False, 5, 1, 64, 1),  # [B, 64, T]
        ConvLayerConfig(False, 3, 64, 64, 2),  # [B, 64, T/2] - downsample
        ConvLayerConfig(False, 3, 64, 64, 1),  # [B, 64, T/2]
        ConvLayerConfig(False, 3, 64, 128, 1),  # [B, 128, T/2] - increase channels
        ConvLayerConfig(False, 3, 128, 128, 1),  # [B, 128, T/2]
        ConvLayerConfig(False, 3, 128, 256, 1),  # [B, 256, T/2]
        AttentionLayerConfig(256, 4),  # [B, 256, T/2] - attention
        ConvLayerConfig(False, 3, 256, 256, 1),  # [B, 256, T/2]
        ConvLayerConfig(True, 3, 256, 128, 2),  # [B, 128, T] - upsample back
        ConvLayerConfig(False, 3, 128, 64, 1),  # [B, 64, T]
        ConvLayerConfig(False, 3, 64, 32, 1),  # [B, 32, T]
        ConvLayerConfig(False, 3, 32, 1, 1),  # [B, 1, T] - final output
    ],
    "discriminator": [
        ConvLayerConfig(False, 3, 1, 16, 2),
        ConvLayerConfig(False, 3, 16, 32, 2),
        ConvLayerConfig(False, 3, 32, 64, 2),
        ConvLayerConfig(False, 3, 64, 128, 2),
        ConvLayerConfig(False, 3, 128, 256, 2),
    ],
}


class LoadPINCoarse(nn.Module):
    """Load-PIN model generator coarse component."""

    def __init__(
        self,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()
        for layer_config in __LOAD_PIN_CONFIG__["coarse"][:-1]:
            if layer_config.transpose:
                # For exact size recovery with odd kernel ConvTranspose1d
                padding = layer_config.kernel_size // 2
                output_padding = layer_config.stride - 1

                conv_layer = nn.ConvTranspose1d(
                    in_channels=layer_config.in_channels,
                    out_channels=layer_config.out_channels,
                    kernel_size=layer_config.kernel_size,
                    stride=layer_config.stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            else:
                conv_layer = nn.Conv1d(
                    in_channels=layer_config.in_channels,
                    out_channels=layer_config.out_channels,
                    kernel_size=layer_config.kernel_size,
                    stride=layer_config.stride,
                    padding=layer_config.kernel_size // 2,
                )
            gate_layer = nn.Conv1d(
                in_channels=layer_config.out_channels,
                out_channels=layer_config.out_channels,
                kernel_size=1,
            )
            self.layers.append(nn.ModuleList([conv_layer, gate_layer]))
        # last layer is just conv
        last_layer_config = __LOAD_PIN_CONFIG__["coarse"][-1]
        if last_layer_config.transpose:
            padding = last_layer_config.kernel_size // 2
            output_padding = last_layer_config.stride - 1

            self.last_layer = nn.ConvTranspose1d(
                in_channels=last_layer_config.in_channels,
                out_channels=last_layer_config.out_channels,
                kernel_size=last_layer_config.kernel_size,
                stride=last_layer_config.stride,
                padding=padding,
                output_padding=output_padding,
            )
        else:
            self.last_layer = nn.Conv1d(
                in_channels=last_layer_config.in_channels,
                out_channels=last_layer_config.out_channels,
                kernel_size=last_layer_config.kernel_size,
                stride=last_layer_config.stride,
                padding=last_layer_config.kernel_size // 2,
            )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, noise_std: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs:
            x: Input tensor of shape (batch_size, seq_len, 1)
            mask: Mask tensor of shape (batch_size, seq_len, 1). 1=observed.
            noise_std: Standard deviation for Gaussian noise injection.

        Outputs:
            x: Output tensor of shape (batch_size, seq_len, 1)
            mu: mean of the original input tensor, used for normalization.
            [batch_size, 1, 1]
            std: standard deviation of the original input tensor, used for normalization. [batch_size, 1, 1]
        """
        B, T, C = x.shape
        x = x * mask  # masking

        # Step 0: Normalize input
        mu = (x * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        )  # [B, 1, 1]
        std = ((x - mu) ** 2 * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        )  # [B, 1, 1]
        std = torch.sqrt(std + 1e-6)  # Avoid division by zero
        x = (x - mu) / std  # Normalize to zero mean, unit variance

        # Add Gaussian noise z to missing positions (common approach in conditional GANs)
        if self.training:
            noise = torch.randn_like(x) * noise_std
            # Only add noise to missing positions, keep observed values intact
            x = x + noise * (1 - mask)

        x = x.permute(0, 2, 1)  # Change to conv shape (B, C, T)
        for conv_layer, gate_layer in self.layers:
            x = conv_layer(x)
            x = self.activation(x)
            gate = nn.Sigmoid()(gate_layer(x))
            x = x * gate
        x = self.last_layer(x)
        x = x.permute(0, 2, 1)  # Change back to (B, T, C)

        # Step -1: Rescale back to original scale
        x = x * std + mu

        return x, mu, std


class LoadPINRefine(nn.Module):
    """Load-PIN model generator refine component."""

    def __init__(
        self,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.attention_layer = None
        self.layers = nn.ModuleList()

        for layer_config in __LOAD_PIN_CONFIG__["refine"][:-1]:
            if isinstance(layer_config, AttentionLayerConfig):
                attn_layer = nn.TransformerEncoderLayer(
                    d_model=layer_config.d_model,
                    nhead=layer_config.nhead,
                    batch_first=True,
                )
                self.layers.append(attn_layer)
            else:
                if layer_config.transpose:
                    # For exact size recovery with odd kernel ConvTranspose1d
                    padding = layer_config.kernel_size // 2
                    output_padding = layer_config.stride - 1

                    conv_layer = nn.ConvTranspose1d(
                        in_channels=layer_config.in_channels,
                        out_channels=layer_config.out_channels,
                        kernel_size=layer_config.kernel_size,
                        stride=layer_config.stride,
                        padding=padding,
                        output_padding=output_padding,
                    )
                else:
                    conv_layer = nn.Conv1d(
                        in_channels=layer_config.in_channels,
                        out_channels=layer_config.out_channels,
                        kernel_size=layer_config.kernel_size,
                        stride=layer_config.stride,
                        padding=layer_config.kernel_size // 2,
                    )
                gate_layer = nn.Conv1d(
                    in_channels=layer_config.out_channels,
                    out_channels=layer_config.out_channels,
                    kernel_size=1,
                )
                self.layers.append(nn.ModuleList([conv_layer, gate_layer]))

        # last layer is just conv
        last_layer_config = __LOAD_PIN_CONFIG__["refine"][-1]
        if last_layer_config.transpose:
            padding = last_layer_config.kernel_size // 2
            output_padding = last_layer_config.stride - 1

            self.last_layer = nn.ConvTranspose1d(
                in_channels=last_layer_config.in_channels,
                out_channels=last_layer_config.out_channels,
                kernel_size=last_layer_config.kernel_size,
                stride=last_layer_config.stride,
                padding=padding,
                output_padding=output_padding,
            )
        else:
            self.last_layer = nn.Conv1d(
                in_channels=last_layer_config.in_channels,
                out_channels=last_layer_config.out_channels,
                kernel_size=last_layer_config.kernel_size,
                stride=last_layer_config.stride,
                padding=last_layer_config.kernel_size // 2,
            )

    def forward(
        self, x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs:
            x: Input tensor of shape (batch_size, seq_len, 1)
            mu: mean of the original input tensor, used for normalization.
            [batch_size, 1, 1]
            std: standard deviation of the original input tensor, used for normalization. [batch_size, 1, 1]

        Outputs:
            x: Output tensor of shape (batch_size, seq_len, 1)
        """
        B, T, C = x.shape  # Store original dimensions

        x = (x - mu) / std  # Normalize to zero mean, unit variance

        x = x.permute(0, 2, 1)  # Change to conv shape (B, C, T)

        conv_idx = 0
        for layer in self.layers:
            if isinstance(layer, nn.TransformerEncoderLayer):
                x = x.permute(0, 2, 1)  # Change to (B, T, C) for attention
                x = layer(x)
                x = x.permute(0, 2, 1)  # Change back to (B, C, T)
            else:
                conv_layer, gate_layer = layer
                x = conv_layer(x)
                x = self.activation(x)
                gate = nn.Sigmoid()(gate_layer(x))
                x = x * gate
                conv_idx += 1

        x = self.last_layer(x)
        x = x.permute(0, 2, 1)  # Change back to (B, T, C)

        x = x * std + mu  # Rescale back to original scale

        return x


class LoadPINDiscriminator(nn.Module):
    """Load-PIN model discriminator.

    Args:
        sequence_length (int): Length of the input sequence.
        This is used to determine the input shape of the fully connected layer.

    """

    def __init__(
        self,
        sequence_length: int,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.activation = nn.ReLU()
        self.conv_layers = nn.ModuleList()
        for layer_config in __LOAD_PIN_CONFIG__["discriminator"]:
            conv_layer = spectral_norm(
                nn.Conv1d(
                    in_channels=layer_config.in_channels,
                    out_channels=layer_config.out_channels,
                    kernel_size=layer_config.kernel_size,
                    stride=layer_config.stride,
                    padding=layer_config.kernel_size // 2,
                )
            )
            self.conv_layers.append(conv_layer)  # shape [B, c, t]
        self.last_layer = spectral_norm(
            nn.Linear(
                in_features=self._get_last_conv_output_shape(),
                out_features=1,
            )
        )

    def _get_last_conv_output_shape(self) -> int:
        """Calculate the flattened size after all conv layers using precise formula."""
        current_length = self.sequence_length

        for layer_config in __LOAD_PIN_CONFIG__["discriminator"]:
            if layer_config.stride > 1:
                # Precise conv1d output size formula
                padding = layer_config.kernel_size // 2
                current_length = (
                    current_length + 2 * padding - layer_config.kernel_size
                ) // layer_config.stride + 1

        # Final shape: [B, out_channels, reduced_length]
        last_conv_layer = self.conv_layers[-1]
        total_features = last_conv_layer.out_channels * current_length

        return total_features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Inputs:
            x: Input tensor of shape (batch_size, seq_len, 1)

        Outputs:
            tuple:
            - (tensor): Realness of the input tensor. shape (batch_size, 1)
            - (list[tensor]): list of features of each conv layer.
        """
        x = x.permute(0, 2, 1)
        features = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.activation(x)
            features.append(x)
        x = x.flatten(start_dim=1)  # [b, c, t] -> [b, c*t]
        realness = self.last_layer(x)  # [b, c*t] -> [b, 1]
        return realness, features


class LoadPINBaseline(NNImputeBaseline):
    """LoadPIN baseline model for imputation tasks."""

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
        noise_std: float = 0.1,
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
            noise_std (float): Standard deviation for Gaussian noise injection.
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
        self.noise_std = noise_std

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.coarse = LoadPINCoarse().to(self.device)
        self.refine = LoadPINRefine().to(self.device)
        self.discriminator = LoadPINDiscriminator(sequence_length).to(self.device)

        # Initialize optimizers
        self.optimizer_coarse = torch.optim.Adam(
            self.coarse.parameters(), lr=self.learning_rate
        )
        self.optimizer_refine = torch.optim.Adam(
            self.refine.parameters(), lr=self.learning_rate
        )
        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )

        self.is_fitted = False

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

    def _loss_coarse(
        self,
        x_gen: torch.Tensor,
        x_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Coarse loss function.
        Args:
            x_gen: Generated tensor by LoadPINCoarse. shape (batch_size, seq_len, 1)
            x_real: Real tensor of shape (batch_size, seq_len, 1)
        Returns:
            RMSE value.
        """
        return torch.sqrt(F.mse_loss(x_gen, x_real))

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
            x_gen: Generated tensor by LoadPINRefine. shape (batch_size, seq_len, 1)
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
            columns=["Time", "Original", "Final", "Status", "Coarse", "Refined"],
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
        ax1.set_title(f"LoadPIN Imputation - Final Result - Epoch {epoch}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Coarse vs Refined generation
        ax2.plot(
            time_steps,
            sample_coarse[:, 0],
            "o-",
            color="orange",
            label="Coarse output",
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

    def _train_coarse_step(
        self,
        x_real: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Train coarse generator for one step.
        Args:
            x_real: Real tensor of shape (batch_size, seq_len, 1)
            mask: Mask tensor of shape (batch_size, seq_len, 1). 1=observed.
        Returns:
            Coarse generator loss.
        """
        self.optimizer_coarse.zero_grad()

        # Generate coarse output
        x_coarse, _mu, _std = self.coarse(x_real, mask, self.noise_std)

        # Compute coarse loss
        coarse_loss = self._loss_coarse(x_coarse, x_real)

        # Backward and optimize
        coarse_loss.backward()
        self.optimizer_coarse.step()

        return coarse_loss

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
            Tuple of (coarse_output, refine_loss).
        """
        self.optimizer_refine.zero_grad()

        # Generate coarse output (detached to prevent gradients flowing back)
        with torch.no_grad():
            x_coarse, mu, std = self.coarse(x_real, mask, self.noise_std)

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
            x_coarse, mu, std = self.coarse(x_real, mask, self.noise_std)
            x_gen = self.refine(x_coarse, mu, std)

        # Compute discriminator loss
        disc_loss = self._loss_discriminator(x_gen, x_real)

        # Backward and optimize
        disc_loss.backward()
        self.optimizer_discriminator.step()

        return disc_loss

    def fit(self, dataloader):
        """
        Fit LoadPIN model on training data using DataLoader.

        Args:
            dataloader: DataLoader that yields (ts_3d_batch, mask_3d_batch)
        """
        # Initialize wandb
        self._initialize_wandb()

        print("Training LoadPIN model...")

        # Print model parameters
        coarse_params = sum(
            p.numel() for p in self.coarse.parameters() if p.requires_grad
        )
        refine_params = sum(
            p.numel() for p in self.refine.parameters() if p.requires_grad
        )
        disc_params = sum(
            p.numel() for p in self.discriminator.parameters() if p.requires_grad
        )
        total_params = coarse_params + refine_params + disc_params

        print(f"LoadPIN model initialized with {total_params:,} trainable parameters")
        print(f"  - Coarse: {coarse_params:,}")
        print(f"  - Refine: {refine_params:,}")
        print(f"  - Discriminator: {disc_params:,}")

        if self.use_wandb:
            wandb.log(
                {
                    "model_parameters_total": total_params,
                    "model_parameters_coarse": coarse_params,
                    "model_parameters_refine": refine_params,
                    "model_parameters_discriminator": disc_params,
                }
            )

        # Training loop
        pbar_epoch = tqdm(range(self.epochs), desc="LoadPIN Training")
        for epoch in pbar_epoch:
            total_coarse_loss = 0
            total_refine_loss = 0
            total_disc_loss = 0
            num_batches = 0

            pbar_batch = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False
            )
            for batch_data, batch_masks in pbar_batch:
                batch_data = batch_data.to(self.device)
                batch_masks = batch_masks.to(self.device)

                # Train coarse generator
                coarse_loss = self._train_coarse_step(batch_data, batch_masks)
                total_coarse_loss += coarse_loss.item()

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
                            "train/batch_coarse_loss": coarse_loss.item(),
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
                        "C": f"{coarse_loss.item():.3f}",
                        "R": f"{refine_loss.item():.3f}",
                        "D": f"{disc_loss.item():.3f}",
                    }
                )

                num_batches += 1

            # Average losses for the epoch
            avg_coarse_loss = total_coarse_loss / num_batches
            avg_refine_loss = total_refine_loss / num_batches
            avg_disc_loss = total_disc_loss / num_batches

            # Update epoch progress bar
            pbar_epoch.set_postfix(
                {
                    "CoarseL": f"{avg_coarse_loss:.3f}",
                    "RefineL": f"{avg_refine_loss:.3f}",
                    "DiscL": f"{avg_disc_loss:.3f}",
                }
            )

            # Log to wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "train/epoch_coarse_loss": avg_coarse_loss,
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
                        sample_coarse, sample_mu, sample_std = self.coarse(
                            sample_data, sample_masks, self.noise_std
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
                print(f"LoadPIN Epoch {epoch + 1}/{self.epochs}")
                print(f"  Coarse Loss: {avg_coarse_loss:.4f}")
                print(f"  Refine Loss: {avg_refine_loss:.4f}")
                print(f"  Disc Loss: {avg_disc_loss:.4f}")

        self.is_fitted = True
        print("LoadPIN training completed.")

        # Close wandb run
        if self.use_wandb:
            wandb.finish()

    def impute(
        self, time_series: torch.Tensor, mask: torch.Tensor, stochastic: bool = False
    ) -> torch.Tensor:
        """
        Impute missing values using trained LoadPIN model.

        Args:
            time_series: Input time series [B, T, D]
            mask: Mask tensor [B, T, D] where 1=observed, 0=missing
            stochastic: If True, adds noise for diverse samples; if False, deterministic

        Returns:
            Imputed time series [B, T, D]
        """
        if not self.is_fitted:
            raise RuntimeError("LoadPIN model must be fitted before imputation")

        # Set models to evaluation mode
        self.coarse.eval()
        self.refine.eval()

        with torch.no_grad():
            time_series = time_series.to(self.device)
            mask = mask.to(self.device)

            # Generate coarse output
            noise_std = self.noise_std if stochastic else 0.0
            x_coarse, mu, std = self.coarse(time_series, mask, noise_std=noise_std)

            # Generate refined output
            x_refined = self.refine(x_coarse, mu, std)

            # Combine observed values with generated missing values
            imputed = mask * time_series + (1 - mask) * x_refined

        return imputed.cpu()
