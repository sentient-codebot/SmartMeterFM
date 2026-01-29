"""Variational Autoencoder Baseline for conditional generation with AdaLN-style conditioning.

This VAE implementation uses Adaptive Layer Normalization (AdaLN) conditioning similar to
the DenoisingTransformer. Instead of simple concatenation, condition embeddings are mapped
to scale and shift parameters that modulate the batch normalization layers in both encoder
and decoder convolution blocks. This provides more effective conditioning by directly
influencing the normalization statistics at each layer.

The conditioning mechanism:
1. Condition embeddings are processed through modulation networks
2. Scale and shift parameters are generated for each conv layer
3. Batch normalization is modulated as: output = norm(x) * (1 + scale) + shift

This approach allows the conditions to adaptively control the feature distributions
at multiple levels of the network hierarchy.
"""

from collections.abc import Callable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ....utils.configuration import TrainConfig
from ...embedders import get_embedder


class AdaLNResidualBlock(nn.Module):
    """1D conv resnet block with AdaLN conditioning"""

    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(
            channels, affine=False
        )  # No learnable params, controlled by AdaLN
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(
            channels, affine=False
        )  # No learnable params, controlled by AdaLN

        # Modulation networks for AdaLN
        self.adaLN_modulation1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, channels * 2),  # scale and shift for bn1
        )
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, channels * 2),  # scale and shift for bn2
        )

    def forward(self, x, condition_emb):
        """
        Args:
            x (torch.Tensor): [batch, channel, length]
            condition_emb (torch.Tensor): [batch, condition_dim]
        Returns:
            activations (torch.Tensor): [batch, channel, length]
        """
        residual = x

        # First conv + AdaLN
        x = self.conv1(x)
        x = self.bn1(x)
        scale1, shift1 = self.adaLN_modulation1(condition_emb).chunk(2, dim=1)
        scale1 = scale1.unsqueeze(-1)  # [batch, channels, 1] for broadcasting
        shift1 = shift1.unsqueeze(-1)
        x = x * (1 + scale1) + shift1
        x = self.relu(x)

        # Second conv + AdaLN
        x = self.conv2(x)
        x = self.bn2(x)
        scale2, shift2 = self.adaLN_modulation2(condition_emb).chunk(2, dim=1)
        scale2 = scale2.unsqueeze(-1)  # [batch, channels, 1] for broadcasting
        shift2 = shift2.unsqueeze(-1)
        x = x * (1 + scale2) + shift2

        x = residual + x  # shape [batch, channel, length]
        out = self.relu(x)
        return out


class Encoder(nn.Module):
    """VAE Encoder with 1D convolutions and AdaLN conditioning for energy profile data.

    Args:
        num_in_channel: Number of input channels (profile features)
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden dimensions for conv layers
    """

    def __init__(
        self,
        num_in_channel: int,
        latent_dim: int = 128,
        hidden_dims: list[int] = None,
        sequence_length: int = 2976,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        self.num_in_channel = num_in_channel
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.condition_dim = latent_dim
        self.sequence_length = sequence_length

        # Condition processing - shared across all layers
        self.condition_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Build conv layers with AdaLN conditioning
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.adaLN_modulations = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        in_channels = num_in_channel
        for h_dim in hidden_dims:
            # Main conv layer
            self.conv_layers.append(
                nn.Conv1d(in_channels, h_dim, kernel_size=4, stride=2, padding=1)
            )
            # BatchNorm without affine parameters (controlled by AdaLN)
            self.bn_layers.append(nn.BatchNorm1d(h_dim, affine=False))
            # AdaLN modulation for this layer
            self.adaLN_modulations.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(latent_dim, h_dim * 2),  # scale and shift
                )
            )
            # Residual block with conditioning
            self.residual_blocks.append(AdaLNResidualBlock(h_dim, latent_dim))
            in_channels = h_dim

        self.leaky_relu = nn.LeakyReLU(0.2)

        # Calculate flattened size after convolutions
        self.final_latent_length = self.sequence_length // (2 ** len(self.hidden_dims))
        self.final_conv_dim = hidden_dims[-1]

        # Final layers for mean and logvar (no concatenation needed now)
        self.fc_mu = nn.Linear(
            self.final_latent_length * self.final_conv_dim, latent_dim
        )
        self.fc_logvar = nn.Linear(
            self.final_latent_length * self.final_conv_dim, latent_dim
        )

    def forward(
        self,
        x: Float[Tensor, "batch channel length"],
        condition_emb: Float[Tensor, "batch condition_dim"],
    ) -> tuple[Float[Tensor, "batch latent_dim"], Float[Tensor, "batch latent_dim"]]:
        if condition_emb.shape[1] != self.latent_dim:
            raise RuntimeError("condition dimension should be equal to latent dim")

        # Process conditions once
        condition_emb = self.condition_proj(condition_emb)  # [batch, condition_dim]

        # Apply convolutions with AdaLN conditioning
        for i, (conv, bn, adaLN_mod, res_block) in enumerate(
            zip(
                self.conv_layers,
                self.bn_layers,
                self.adaLN_modulations,
                self.residual_blocks,
                strict=True,
            )
        ):
            # Conv layer
            x = conv(x)
            # BatchNorm + AdaLN modulation
            x = bn(x)
            scale, shift = adaLN_mod(condition_emb).chunk(2, dim=1)
            scale = scale.unsqueeze(-1)  # [batch, channels, 1] for broadcasting
            shift = shift.unsqueeze(-1)
            x = x * (1 + scale) + shift
            x = self.leaky_relu(x)
            # Residual block with conditioning
            x = res_block(x, condition_emb)

        # disabled: Global average pooling to get fixed size representation
        # x = torch.mean(x, dim=-1)  # [batch, final_conv_dim]
        x = rearrange(x, "B C L -> B (C L)")

        # Compute mean and logvar (no concatenation needed)
        mu = self.fc_mu(x)  # (B, latent_dim)
        logvar = self.fc_logvar(x)  # (B, latent_dim)

        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder with 1D transposed convolutions and AdaLN conditioning.

    Args:
        latent_dim: Dimension of latent space
        num_out_channel: Number of output channels (profile features)
        output_length: Length of output sequence
        hidden_dims: List of hidden dimensions for deconv layers (reversed from encoder)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_out_channel: int = 1,
        output_length: int = 2976,  # 31 days * 96 steps/day
        hidden_dims: list[int] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        self.latent_dim = latent_dim
        self.num_out_channel = num_out_channel
        self.output_length = output_length
        self.condition_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Calculate initial feature map size
        self.num_downsamples = len(hidden_dims)
        self.init_length = output_length // (2**self.num_downsamples)
        self.init_channels = hidden_dims[0]

        # Condition processing - shared across all layers
        self.condition_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Initial projection from latent to feature maps (no condition concatenation)
        self.fc_decode = nn.Linear(latent_dim, self.init_channels * self.init_length)

        # Build deconv layers with AdaLN conditioning
        self.deconv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.adaLN_modulations = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            # Deconv layer
            self.deconv_layers.append(
                nn.ConvTranspose1d(
                    in_channels, h_dim, kernel_size=4, stride=2, padding=1
                )
            )
            # BatchNorm without affine parameters (controlled by AdaLN)
            self.bn_layers.append(nn.BatchNorm1d(h_dim, affine=False))
            # AdaLN modulation for this layer
            self.adaLN_modulations.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(latent_dim, h_dim * 2),  # scale and shift
                )
            )
            # Residual block with conditioning
            self.residual_blocks.append(AdaLNResidualBlock(h_dim, latent_dim))
            in_channels = h_dim

        self.relu = nn.ReLU()

        # Final layer to output channels
        self.final_deconv = nn.ConvTranspose1d(
            in_channels, num_out_channel, kernel_size=4, stride=2, padding=1
        )

        # Final adjustment layer to get exact output length
        self.final_adjust = nn.Conv1d(num_out_channel, num_out_channel, kernel_size=1)

    def forward(
        self,
        z: Float[Tensor, "batch latent_dim"],
        condition_emb: Float[Tensor, "batch latent_dim"],
    ) -> Float[Tensor, "batch num_out_channel output_length"]:
        # Process conditions once
        condition_emb = self.condition_proj(condition_emb)

        # Project latent to feature map (no concatenation)
        x = self.fc_decode(z)
        x = x.view(-1, self.init_channels, self.init_length)

        # Apply deconvolutions with AdaLN conditioning
        for i, (deconv, bn, adaLN_mod, res_block) in enumerate(
            zip(
                self.deconv_layers,
                self.bn_layers,
                self.adaLN_modulations,
                self.residual_blocks, strict=False,
            )
        ):
            # Deconv layer
            x = deconv(x)
            # BatchNorm + AdaLN modulation
            x = bn(x)
            scale, shift = adaLN_mod(condition_emb).chunk(2, dim=1)
            scale = scale.unsqueeze(-1)  # [batch, channels, 1] for broadcasting
            shift = shift.unsqueeze(-1)
            x = x * (1 + scale) + shift
            x = self.relu(x)
            # Residual block with conditioning
            x = res_block(x, condition_emb)

        # Final deconv to output channels
        x = self.final_deconv(x)

        # Adjust to exact output length if needed
        if x.shape[-1] != self.output_length:
            raise RuntimeError(
                f"output length mismatch {x.shape[-1]}!={self.output_length}"
            )

        x = self.final_adjust(x)
        return x


class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder for energy profile generation.

    Args:
        num_in_channel: Number of input/output channels
        latent_dim: Dimension of latent space
        condition_dim: Dimension of condition embedding
        output_length: Length of output sequence
        hidden_dims: Hidden dimensions for encoder (decoder uses reversed)
    """

    def __init__(
        self,
        num_in_channel: int,
        latent_dim: int = 128,
        output_length: int = 2976,
        hidden_dims: list[int] = None,
    ):
        super().__init__()

        self.num_in_channel = num_in_channel
        self.latent_dim = latent_dim
        self.condition_dim = latent_dim
        self.output_length = output_length

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.encoder = Encoder(
            num_in_channel=num_in_channel,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )

        decoder_hidden_dims = list(reversed(hidden_dims))
        self.decoder = Decoder(
            latent_dim=latent_dim,
            num_out_channel=num_in_channel,
            output_length=output_length,
            hidden_dims=decoder_hidden_dims,
        )

    def encode(self, x: Tensor, condition_emb: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(x, condition_emb)

    def decode(self, z: Tensor, condition_emb: Tensor) -> Tensor:
        return self.decoder(z, condition_emb)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: Tensor, condition_emb: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x, condition_emb)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition_emb)
        return recon, mu, logvar

    def sample(self, condition_emb: Tensor, num_samples: int = 1) -> Tensor:
        """Generate samples from the VAE."""
        batch_size = condition_emb.shape[0]
        device = condition_emb.device

        if num_samples > 1:
            # Expand conditions for multiple samples per condition
            condition_emb = condition_emb.repeat_interleave(num_samples, dim=0)
            z = torch.randn(batch_size * num_samples, self.latent_dim, device=device)
        else:
            z = torch.randn(batch_size, self.latent_dim, device=device)

        return self.decode(z, condition_emb)


class VAEModelPL(pl.LightningModule):
    """PyTorch Lightning wrapper for Conditional VAE baseline.

    This serves as a baseline model for conditional energy profile generation,
    comparable to the FlowModelPL but using VAE instead of flow matching.
    """

    def __init__(
        self,
        num_in_channel: int,
        train_config: TrainConfig,
        latent_dim: int = 128,
        output_length: int = 2976,
        hidden_dims: list[int] = None,
        beta: float = 1.0,
        label_embedder_name: str | None = None,
        label_embedder_args: dict | None = None,
        metrics_factory: Callable | None = None,
        create_mask: bool = False,
    ):
        super().__init__()

        # Initialize embedders (similar to FlowModelPL)
        label_embedder = None
        if label_embedder_name is not None:
            label_embedder_args = label_embedder_args or {}
            label_embedder = get_embedder(label_embedder_name, **label_embedder_args)

        self.label_embedder = label_embedder
        # self.context_embedder = context_embedder
        # NOTE: cond vae does not need context embedder. it's for cross attention.

        # VAE model
        self.vae = ConditionalVAE(
            num_in_channel=num_in_channel,
            latent_dim=latent_dim,
            output_length=output_length,
            hidden_dims=hidden_dims,
        )

        # Store config
        self.num_in_channel = num_in_channel
        self.train_config = train_config
        self.beta = beta  # Beta for beta-VAE
        self.create_mask = create_mask
        self.condition_dim = latent_dim

        # Metrics
        self.metrics = metrics_factory() if metrics_factory is not None else None

        self.save_hyperparameters()

    def _format_input_y(
        self,
        which: str,
        batch_size: int,
        device: torch.device,
        y: dict | None,
        dict_extras: dict | None,
    ):
        if which == "label":
            embedder = self.label_embedder
        elif which == "context":
            embedder = self.context_embedder
        else:
            raise ValueError("`which` must be label or context")
        assert embedder is not None, f"{which} embedder is None"

        force_drop_ids = torch.ones(
            (batch_size,), device=device, dtype=torch.long
        )  # shape: (batch, )

        # fill in missing keys
        if dict_extras is None:
            dict_extras = {}

        for label_name in embedder.embedder_keys:
            if label_name not in dict_extras:
                dict_extras[label_name] = {}

        if y is None:
            y = {}
            "if y is None, then seen as force drop all conditions"
        for label_name in getattr(embedder, "embedder_keys", []):
            if label_name not in y:
                y[label_name] = None
                dict_extras[label_name]["force_drop_ids"] = force_drop_ids

        return y, dict_extras

    def _get_condition_embedding(self, condition: dict[str, Tensor]) -> Tensor:
        """Process conditions using embedders to get condition embedding."""
        if self.label_embedder is not None:
            # Use the same formatting as DenoisingTransformer
            batch_size = next(iter(condition.values())).shape[0] if condition else 1
            device = (
                next(iter(condition.values())).device
                if condition
                else next(self.parameters()).device
            )

            _label_y, _dict_emb_extras = self._format_input_y(
                "label",
                batch_size,
                device,
                condition,
                None,
            )
            return self.label_embedder(
                _label_y,
                dict_extra=_dict_emb_extras,
                batch_size=batch_size,
                device=device,
            )
        else:
            raise AttributeError("label_embedder not provided")

    @staticmethod
    def _convert_offset_month_length(month_length: int | Tensor, offset: int):
        """Convert month length with offset (same as FlowModelPL)."""
        return (month_length + offset) * 96

    @staticmethod
    def _create_loss_mask(valid_length: Tensor, full_length: int) -> Tensor:
        """Create a mask for the loss function (same as FlowModelPL)."""
        valid_length = valid_length.long()
        batch_size = valid_length.shape[0]
        loss_mask = torch.zeros(batch_size, full_length)
        for i in range(batch_size):
            loss_mask[i, : valid_length[i]] = 1.0
        return loss_mask

    def vae_loss(
        self,
        recon: Tensor,
        target: Tensor,
        mu: Tensor,
        logvar: Tensor,
        loss_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute VAE loss with optional masking."""

        if loss_mask is not None:
            # Apply mask to reconstruction loss
            recon_loss = F.mse_loss(
                recon * loss_mask, target * loss_mask, reduction="sum"
            )
            recon_loss = recon_loss / loss_mask.sum()
        else:
            recon_loss = F.mse_loss(recon, target)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = torch.clamp(kl_loss, min=0.0)
        kl_loss = kl_loss / target.shape[0]  # Normalize by batch size

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return {"total_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def configure_optimizers(self):
        params = list(self.vae.parameters())
        if self.label_embedder is not None:
            params.extend(self.label_embedder.parameters())
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.train_config.lr,
            betas=self.train_config.adam_betas,
        )
        return optimizer

    def training_step(
        self,
        batch: tuple[Tensor, dict[str, Tensor]],
        batch_idx: int,
    ):
        self.train()
        profile, condition = batch  # profile shape [B, L, D]

        # Get condition embedding
        condition_emb = self._get_condition_embedding(condition)

        # Forward pass through VAE
        recon, mu, logvar = self.vae(
            rearrange(profile, "B L D -> B 1 (L D)"), condition_emb
        )
        recon = rearrange(recon, "B 1 (L D) -> B L D", D=profile.shape[-1])

        # Handle masking if needed
        loss_mask = None
        if self.create_mask and "month_length" in condition:
            valid_length = self._convert_offset_month_length(
                condition["month_length"], 28
            ).squeeze(1)
            loss_mask = self._create_loss_mask(
                valid_length=valid_length,
                full_length=profile.shape[1] * profile.shape[2],
            )
            loss_mask = loss_mask.to(profile.device)
            loss_mask = rearrange(
                loss_mask,
                "batch (sequence channel) -> batch sequence channel",
                sequence=profile.shape[1],
            )

        # Compute loss
        losses = self.vae_loss(recon, profile, mu, logvar, loss_mask)

        # Log losses
        self.log(
            "Train/total_loss",
            losses["total_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "Train/recon_loss",
            losses["recon_loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Train/kl_loss",
            losses["kl_loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # Monitor posterior collapse by logging std of mu across batch dimension
        mu_std = torch.std(mu, dim=0).mean()  # Average std across latent dimensions
        self.log(
            "Train/mu_std",
            mu_std,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return losses["total_loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        self.eval()
        profile, condition = batch

        # Get condition embedding
        condition_emb = self._get_condition_embedding(condition)

        # Forward pass through VAE
        recon, mu, logvar = self.vae(
            rearrange(profile, "B L D -> B 1 (L D)"), condition_emb
        )
        recon = rearrange(recon, "B 1 (L D) -> B L D", D=profile.shape[-1])

        # Handle masking if needed
        loss_mask = None
        if self.create_mask and "month_length" in condition:
            valid_length = self._convert_offset_month_length(
                condition["month_length"], 28
            ).squeeze(1)
            loss_mask = self._create_loss_mask(
                valid_length=valid_length,
                full_length=profile.shape[1] * profile.shape[2],
            )
            loss_mask = loss_mask.to(profile.device)
            loss_mask = rearrange(
                loss_mask,
                "batch (sequence channel) -> batch sequence channel",
                sequence=profile.shape[1],
            )

        # Compute loss
        losses = self.vae_loss(recon, profile, mu, logvar, loss_mask)

        # Log validation losses
        self.log(
            "Validation/total_loss",
            losses["total_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "Validation/recon_loss",
            losses["recon_loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Validation/kl_loss",
            losses["kl_loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Monitor posterior collapse by logging std of mu across batch dimension
        mu_std = torch.std(mu, dim=0).mean()  # Average std across latent dimensions
        self.log(
            "Validation/mu_std",
            mu_std,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return None

    @torch.no_grad()
    def sample(
        self,
        condition: dict[str, Tensor],
        num_samples: int = 1,
        return_latents: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample from the VAE given conditions.

        Args:
            condition: Dictionary of condition tensors
            num_samples: Number of samples to generate per condition
            return_latents: Whether to return the latent codes used

        Returns:
            Generated samples, optionally with latent codes. shape [bs_cond * num_samples]   asdf
        """
        self.eval()

        # Get condition embedding
        condition_emb = self._get_condition_embedding(condition)

        # Generate samples
        samples = self.vae.sample(condition_emb, num_samples)

        if return_latents:
            # Also return the latent codes used
            batch_size = condition_emb.shape[0]
            device = condition_emb.device

            if num_samples > 1:
                z = torch.randn(
                    batch_size * num_samples, self.vae.latent_dim, device=device
                )
            else:
                z = torch.randn(batch_size, self.vae.latent_dim, device=device)

            return samples, z

        return samples

    def encode_data(
        self, profile: Tensor, condition: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Encode data to latent space.

        Args:
            profile: Energy profile data
            condition: Condition dictionary

        Returns:
            tuple of (mu, logvar) from encoder
        """
        self.eval()

        condition_emb = self._get_condition_embedding(condition)
        with torch.no_grad():
            mu, logvar = self.vae.encode(profile, condition_emb)

        return mu, logvar
