"""Generative Adversarial Network Baseline for conditional generation with AdaLN-style conditioning.

This GAN implementation uses Adaptive Layer Normalization (AdaLN) conditioning similar to
the DenoisingTransformer and VAE baseline. The Generator uses AdaLN to modulate batch
normalization layers, while the Discriminator processes both real/fake data and condition
information to provide better training signals.

The conditioning mechanism:
1. Condition embeddings are processed through modulation networks
2. Scale and shift parameters are generated for each conv layer in Generator
3. Batch normalization is modulated as: output = norm(x) * (1 + scale) + shift
4. Discriminator uses condition information to distinguish real from fake samples

This approach allows the conditions to adaptively control the feature distributions
at multiple levels of the network hierarchy, similar to the VAE baseline.
"""

import copy
from collections.abc import Callable
from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from ....utils.configuration import TrainConfig
from ...embedders import get_embedder


class AdaLNResidualBlock(nn.Module):
    """1D conv ResNet block with AdaLN conditioning for Generator.

    Note: Use LayerNorm (per-position over channels) instead of BatchNorm to
    avoid batch-dependent behavior and better handle variable batch sizes.

    Args:
        - channels: number of in feature channels
        - condition_dim: conditioning vector dim
        - z_dim: optional conditioning vector from noise vector z
    """

    def __init__(self, channels: int, condition_dim: int, z_dim: int = 0):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        # LayerNorm over channels at each position (affine=False; AdaLN provides scale/shift)
        self.ln1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(channels, elementwise_affine=False)

        # Modulation networks for AdaLN
        self.adaLN_modulation1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim + z_dim, channels * 2),  # scale and shift for bn1
        )
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim + z_dim, channels * 2),  # scale and shift for bn2
        )

    def forward(self, x, condition_emb, z: Tensor | None = None):
        """
        Args:
            x (torch.Tensor): [batch, channel, length]
            condition_emb (torch.Tensor): [batch, condition_dim]
            z (torch.Tensor|None): [batch, z_dim] or None
        Returns:
            activations (torch.Tensor): [batch, channel, length]
        """
        residual = x

        if z is not None:
            condition_emb = torch.cat(
                [condition_emb, z], dim=-1
            )  # concat two cond vectors

        # First conv + AdaLN
        x = self.conv1(x)
        # Apply LayerNorm over channel dimension for each position
        x = rearrange(x, "B C L -> B L C")
        x = self.ln1(x)
        x = rearrange(x, "B L C -> B C L")
        scale1, shift1 = self.adaLN_modulation1(condition_emb).chunk(2, dim=1)
        scale1 = scale1.unsqueeze(-1)  # [batch, channels, 1] for broadcasting
        shift1 = shift1.unsqueeze(-1)
        x = x * (1 + scale1) + shift1
        x = self.relu(x)

        # Second conv + AdaLN
        x = self.conv2(x)
        x = rearrange(x, "B C L -> B L C")
        x = self.ln2(x)
        x = rearrange(x, "B L C -> B C L")
        scale2, shift2 = self.adaLN_modulation2(condition_emb).chunk(2, dim=1)
        scale2 = scale2.unsqueeze(-1)  # [batch, channels, 1] for broadcasting
        shift2 = shift2.unsqueeze(-1)
        x = x * (1 + scale2) + shift2

        x = residual + x  # shape [batch, channel, length]
        out = self.relu(x)
        return out


class Generator(nn.Module):
    """GAN Generator with 1D transposed convolutions and AdaLN conditioning.

    Args:
        latent_dim: Dimension of latent noise vector
        condition_dim: Dimension of condition embedding
        num_out_channel: Number of output channels (profile features)
        output_length: Length of output sequence
        hidden_dims: List of hidden dimensions for deconv layers

    Input/output shape: [batch, output_length, num_out_channel]
    """

    def __init__(
        self,
        latent_dim: int = 128,
        condition_dim: int = 128,
        num_out_channel: int = 1,
        output_length: int = 2976 // 1,
        hidden_dims: list[int] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_out_channel = num_out_channel
        self.output_length = output_length
        self.hidden_dims = hidden_dims

        # Calculate initial feature map size
        self.num_downsamples = len(hidden_dims)
        flat_length = num_out_channel * output_length
        assert flat_length % (2**self.num_downsamples) == 0, (
            "flat_length must be divisible by 2 ** len(hidden_dims)"
        )
        self.init_length = flat_length // (2**self.num_downsamples)
        self.init_channels = hidden_dims[0]

        # Condition processing - shared across all layers
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        # Initial projection from noise + condition to feature maps
        self.fc_decode = nn.Linear(
            latent_dim + condition_dim, self.init_channels * self.init_length
        )

        # Build deconv layers with AdaLN conditioning
        self.deconv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.adaLN_modulations = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        self.noise_strengths = nn.ParameterList()

        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            # Deconv layer
            self.deconv_layers.append(
                nn.ConvTranspose1d(
                    in_channels, h_dim, kernel_size=4, stride=2, padding=1
                )
            )
            # LayerNorm keeps the activations centered per position
            self.norm_layers.append(nn.LayerNorm(h_dim, elementwise_affine=False))
            # AdaLN modulation for this layer
            self.adaLN_modulations.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(condition_dim + latent_dim, h_dim * 2),  # scale and shift
                )
            )
            # Residual block with conditioning
            self.residual_blocks.append(
                AdaLNResidualBlock(h_dim, condition_dim, latent_dim)
            )
            self.noise_strengths.append(nn.Parameter(torch.zeros(1)))
            in_channels = h_dim

        self.relu = nn.ReLU()

        # Final layer outputs a single channel sequence of length L*D
        self.final_deconv = nn.ConvTranspose1d(
            in_channels, 1, kernel_size=4, stride=2, padding=1
        )

        # Final 1x1 conv over the single sequence channel
        self.final_adjust = nn.Conv1d(1, 1, kernel_size=1)

    def forward(
        self,
        z: Float[Tensor, "batch latent_dim"],
        condition_emb: Float[Tensor, "batch condition_dim"],
    ) -> Float[Tensor, "batch seq_len data_dim"]:
        # Process conditions once
        condition_emb_processed = self.condition_proj(condition_emb)

        # Combine noise and condition
        z_cond = torch.cat([z, condition_emb_processed], dim=1)

        # Project to feature map
        x = self.fc_decode(z_cond)
        x = x.view(-1, self.init_channels, self.init_length)

        # Apply deconvolutions with AdaLN conditioning
        for deconv, norm, adaLN_mod, res_block, noise_strength in zip(
            self.deconv_layers,
            self.norm_layers,
            self.adaLN_modulations,
            self.residual_blocks,
            self.noise_strengths,
            strict=True,
        ):
            # Deconv layer
            x = deconv(x)
            if self.training:
                x = x + noise_strength * torch.randn_like(x)
            # LayerNorm + AdaLN modulation
            x = rearrange(x, "B C L -> B L C")
            x = norm(x)
            x = rearrange(x, "B L C -> B C L")
            scale, shift = adaLN_mod(z_cond).chunk(2, dim=1)
            scale = scale.unsqueeze(-1)  # [batch, channels, 1] for broadcasting
            shift = shift.unsqueeze(-1)
            x = x * (1 + scale) + shift
            x = self.relu(x)
            # Residual block with conditioning
            x = res_block(x, condition_emb_processed, z)

        # Final deconv to 1 channel over flattened sequence
        x = self.final_deconv(x)
        # Check exact output length
        if x.shape[-1] != self.output_length * self.num_out_channel:
            raise RuntimeError(
                f"output length mismatch {x.shape[-1]}!={self.output_length * self.num_out_channel}"
            )

        x = self.final_adjust(x)
        # Map internal [B, 1, L*D] to external [B, L, D]
        x = torch.tanh(x)
        x = rearrange(x, "B 1 (L D) -> B L D", D=self.num_out_channel)
        return x


class Discriminator(nn.Module):
    """GAN Discriminator that can process both data and condition information.

    Conditioning is integrated via AdaIN-style modulation: after normalization,
    activations are modulated as x = norm(x) * (1 + scale) + shift, where
    scale/shift are produced from the condition embedding per layer.

    Args:
        num_in_channel: Number of input channels (profile features)
        condition_dim: Dimension of condition embedding
        input_length: Length of input sequence
        hidden_dims: List of hidden dimensions for conv layers
    """

    def __init__(
        self,
        num_in_channel: int = 1,
        condition_dim: int = 128,
        input_length: int = 2976 // 1,
        hidden_dims: list[int] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        # External data dimensions: [B, L, D] with L * D == input_length
        self.num_in_channel = num_in_channel
        self.condition_dim = condition_dim
        self.input_length = input_length
        self.hidden_dims = hidden_dims
        flat_length = input_length * num_in_channel  # flattened input length
        assert flat_length % (2 ** len(self.hidden_dims)) == 0, (
            "flattened input length must be divisible by 2 ** len(hidden_dims)"
        )

        # Condition processing
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(condition_dim, condition_dim),
        )

        # Build conv layers (operate on internal [B, 1, L*D])
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        # Per-layer AdaIN-style modulation from condition
        self.ada_modulations = nn.ModuleList()
        self.use_minibatch_std = True

        in_channels = 1
        for h_dim in hidden_dims:
            # Main conv layer
            conv = nn.Conv1d(in_channels, h_dim, kernel_size=4, stride=2, padding=1)
            self.conv_layers.append(spectral_norm(conv))
            # Instance norm (works better than batch norm for discriminator)
            # affine=False so that AdaIN provides scale/shift when conditioning is used
            self.norm_layers.append(nn.InstanceNorm1d(h_dim, affine=False))
            # AdaIN modulation network for this layer
            self.ada_modulations.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(condition_dim, h_dim * 2),  # scale and shift
                )
            )
            in_channels = h_dim

        self.leaky_relu = nn.LeakyReLU(0.2)

        # Calculate flattened size after convolutions
        self.final_length = flat_length // (2 ** len(self.hidden_dims))
        extra_channels = 1 if self.use_minibatch_std else 0
        self.final_conv_dim = hidden_dims[-1] + extra_channels

        # Classification head (combines data features and condition)
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(self.final_length * self.final_conv_dim, 512)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(256, 1)),
        )

    def forward(
        self,
        x: Float[Tensor, "batch seq_len data_dim"],
        condition_emb: Float[Tensor, "batch condition_dim"] | None = None,
    ) -> Float[Tensor, "batch 1"]:
        """Forward of the discriminator.

        If ``condition_emb`` is provided, apply per-layer AdaIN-style modulation
        after normalization. If ``condition_emb`` is None, behaves as an
        unconditional discriminator (backward compatible with existing calls).
        """
        # External input [B, L, D] -> internal [B, 1, L*D]
        assert x.ndim == 3, "Discriminator expects [B, L, D] input"
        assert x.shape[1] * x.shape[2] == self.input_length * self.num_in_channel, (
            "L * D must equal input_length"
        )
        x = rearrange(x, "B L D -> B 1 (L D)")
        # Process conditions if provided
        condition_emb_processed = (
            self.condition_proj(condition_emb) if condition_emb is not None else None
        )

        # Apply convolutions
        for i, (conv, norm, ada_mod) in enumerate(
            zip(self.conv_layers, self.norm_layers, self.ada_modulations, strict=True)
        ):
            x = conv(x)
            if i > 0:  # Skip normalization for first layer
                x = norm(x)
                if condition_emb_processed is not None:
                    scale, shift = ada_mod(condition_emb_processed).chunk(2, dim=1)
                    scale = scale.unsqueeze(-1)  # [B, C, 1]
                    shift = shift.unsqueeze(-1)
                    x = x * (1 + scale) + shift
            x = self.leaky_relu(x)

        if self.use_minibatch_std:
            batch_size, _, seq_len = x.shape
            if batch_size > 1:
                mean = x.mean(dim=0, keepdim=True)
                variance = torch.mean((x - mean) ** 2, dim=0, keepdim=True)
                std = torch.sqrt(variance + 1e-8)
                mean_std = std.mean().view(1, 1, 1).expand(batch_size, 1, seq_len)
            else:
                mean_std = torch.zeros(
                    batch_size, 1, x.shape[-1], device=x.device, dtype=x.dtype
                )
            x = torch.cat([x, mean_std], dim=1)

        # Flatten data features
        x_flat = rearrange(x, "B C L -> B (C L)")

        # Combine data features with condition (kept simple; classifier remains data-only)

        # Classification
        out = self.classifier(x_flat)  # shape [batch, 1]
        return out


class ConditionalGAN(nn.Module):
    """Conditional Generative Adversarial Network for energy profile generation.

    Args:
        latent_dim: Dimension of noise vector
        condition_dim: Dimension of condition embedding
        num_channel: Number of input/output channels
        sequence_length: Length of input/output sequence
        gen_hidden_dims: Hidden dimensions for generator
        disc_hidden_dims: Hidden dimensions for discriminator

    interally treats the input/output data as [B, 1, num_channel * sequence_length]
    """

    def __init__(
        self,
        latent_dim: int = 128,
        condition_dim: int = 128,
        num_channel: int = 1,
        sequence_length: int = 2976 // 1,
        gen_hidden_dims: list[int] = None,
        disc_hidden_dims: list[int] = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_channel = num_channel
        self.sequence_length = sequence_length

        if gen_hidden_dims is None:
            gen_hidden_dims = [512, 256, 128, 64]
        if disc_hidden_dims is None:
            disc_hidden_dims = [64, 128, 256, 512]

        self.generator = Generator(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            num_out_channel=num_channel,
            output_length=sequence_length,
            hidden_dims=gen_hidden_dims,
        )

        self.discriminator = Discriminator(
            num_in_channel=num_channel,
            condition_dim=condition_dim,
            input_length=sequence_length,
            hidden_dims=disc_hidden_dims,
        )

    def generate(
        self,
        condition_emb: Tensor,
        num_samples: int = 1,
        z: Tensor | None = None,
    ) -> Float[Tensor, "B seq_length num_channel"]:
        """Generate samples from the GAN.

        Returns [B, L, D] externally. If `z` is provided, it is used.
        """
        batch_size = condition_emb.shape[0]
        device = condition_emb.device

        if num_samples > 1:
            # Expand conditions for multiple samples per condition
            condition_emb = condition_emb.repeat_interleave(num_samples, dim=0)
            if z is None:
                z = torch.randn(
                    batch_size * num_samples, self.latent_dim, device=device
                )
        else:
            if z is None:
                z = torch.randn(batch_size, self.latent_dim, device=device)

        return self.generator(z, condition_emb)


class GANModelPL(pl.LightningModule):
    """PyTorch Lightning wrapper for Conditional GAN baseline.

    This serves as a baseline model for conditional energy profile generation,
    comparable to the FlowModelPL and VAEModelPL but using GAN instead.
    """

    def __init__(
        self,
        num_in_channel: int,
        train_config: TrainConfig,
        latent_dim: int = 128,
        output_length: int = 2976 // 16,
        gen_hidden_dims: list[int] = None,
        disc_hidden_dims: list[int] = None,
        label_embedder_name: str | None = None,
        label_embedder_args: dict | None = None,
        metrics_factory: Callable | None = None,
        create_mask: bool = False,
        # GAN-specific hyperparameters
        paradigm: Literal["vanilla", "wgan"] = "vanilla",
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        beta1: float = 0.0,
        beta2: float = 0.9,
        lambda_gp: float = 10.0,  # Gradient penalty coefficient
        d_steps_per_g: int = 2,  # Discriminator steps per one generator step
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: Literal["norm", "value"] = "norm",
    ):
        super().__init__()

        # Initialize embedders (similar to FlowModelPL)
        label_embedder = None
        if label_embedder_name is not None:
            label_embedder_args = label_embedder_args or {}
            label_embedder = get_embedder(label_embedder_name, **label_embedder_args)

        # Create separate embedders for generator and discriminator so they can
        # adapt independently without fighting over shared parameters.
        self.generator_label_embedder = label_embedder
        self.discriminator_label_embedder = (
            copy.deepcopy(label_embedder) if label_embedder is not None else None
        )
        # Backward compatibility for any existing accessors.
        self.label_embedder = self.generator_label_embedder

        # Ensure embedder dim matches latent dim
        if self.generator_label_embedder is not None:
            if not hasattr(self.generator_label_embedder, "dim_embedding"):
                raise RuntimeError("label_embedder must expose dim_embedding")
            assert self.generator_label_embedder.dim_embedding == latent_dim, (
                "latent_dim must equal label embedder dim_embedding"
            )
            if self.discriminator_label_embedder is not None:
                assert (
                    self.discriminator_label_embedder.dim_embedding
                    == self.generator_label_embedder.dim_embedding
                ), "generator and discriminator embedders must share dim_embedding"

        # GAN model
        self.paradigm = paradigm
        self.gan = ConditionalGAN(
            latent_dim=latent_dim,
            condition_dim=latent_dim,
            num_channel=num_in_channel,
            sequence_length=output_length,
            gen_hidden_dims=gen_hidden_dims,
            disc_hidden_dims=disc_hidden_dims,
        )

        # Store config
        self.num_in_channel = num_in_channel
        self.train_config = train_config
        self.create_mask = create_mask
        self.condition_dim = latent_dim

        # GAN hyperparameters
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_gp = lambda_gp
        self.d_steps_per_g = d_steps_per_g
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # Metrics
        self.metrics = metrics_factory() if metrics_factory is not None else None

        # Automatic optimization disabled for GAN training
        self.automatic_optimization = False

        self.save_hyperparameters()

    def _clip_optimizer_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Apply gradient clipping when manual optimization is enabled."""
        if not self.gradient_clip_val or self.gradient_clip_val <= 0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
        )

    def _format_input_y(
        self,
        *,
        embedder: nn.Module,
        batch_size: int,
        device: torch.device,
        y: dict | None,
        dict_extras: dict | None,
    ):
        assert embedder is not None, "embedder is None"

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

    def _get_condition_embedding(
        self,
        condition: dict[str, Tensor],
        *,
        embedder: nn.Module | None,
    ) -> Tensor:
        """Process conditions using embedders to get condition embedding."""
        if embedder is None:
            raise AttributeError("label_embedder not provided")

        # Use the same formatting as DenoisingTransformer
        batch_size = next(iter(condition.values())).shape[0] if condition else 1
        device = (
            next(iter(condition.values())).device
            if condition
            else next(self.parameters()).device
        )

        _label_y, _dict_emb_extras = self._format_input_y(
            embedder=embedder,
            batch_size=batch_size,
            device=device,
            y=condition,
            dict_extras=None,
        )
        return embedder(
            _label_y,
            dict_extra=_dict_emb_extras,
            batch_size=batch_size,
            device=device,
        )

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

    def gradient_penalty(self, real_samples, fake_samples, condition_emb):
        """Calculate gradient penalty for WGAN-GP."""
        batch_size = real_samples.shape[0]
        device = real_samples.device

        # Ensure the fake samples are detached from generator graph
        fake_samples = fake_samples.detach()

        # Random interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1, device=device)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)

        # Get discriminator output for interpolated samples in fp32 for stability
        device_type = interpolated.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            interpolated_fp32 = interpolated.to(dtype=torch.float32)
            condition_fp32 = (
                condition_emb.to(dtype=torch.float32)
                if condition_emb is not None
                else None
            )
            d_interpolated = self.gan.discriminator(interpolated_fp32, condition_fp32)

        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated_fp32,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            only_inputs=True,
        )[0]

        # Calculate gradient penalty
        gradients = gradients.reshape(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        return gradient_penalty

    def configure_optimizers(self):
        # Each network optimizes its own embedder copy.
        gen_params = list(self.gan.generator.parameters())
        if self.generator_label_embedder is not None:
            gen_params.extend(self.generator_label_embedder.parameters())

        disc_params = list(self.gan.discriminator.parameters())
        if self.discriminator_label_embedder is not None:
            disc_params.extend(self.discriminator_label_embedder.parameters())

        opt_g = torch.optim.Adam(
            gen_params,
            lr=self.generator_lr,
            betas=(self.beta1, self.beta2),
        )
        opt_d = torch.optim.Adam(
            disc_params,
            lr=self.discriminator_lr,
            betas=(self.beta1, self.beta2),
        )
        return [opt_g, opt_d], []

    def training_step(self, batch: tuple[Tensor, dict[str, Tensor]], batch_idx: int):
        "two branches: vanilla and wgan"
        profile, condition = batch
        # Each network owns its own embedder so compute condition encodings per owner.
        condition_emb_gen = self._get_condition_embedding(
            condition, embedder=self.generator_label_embedder
        )
        disc_embedder = (
            self.discriminator_label_embedder or self.generator_label_embedder
        )
        condition_emb_disc = self._get_condition_embedding(
            condition, embedder=disc_embedder
        )

        # Validate profile shape
        assert profile.ndim == 3, "profile must be [B, L, D]"
        B, L, D = profile.shape
        assert L == self.gan.sequence_length, (
            f"input length mismatch. model expects {self.gan.sequence_length}, but got {L}. full input shape {profile.shape}"
        )
        assert D == self.gan.num_channel, (
            f"input channel mismatch. model expects {self.gan.num_channel}, but got {D}. full input shape {profile.shape}"
        )

        if self.paradigm == "vanilla":
            return self._training_step_vanilla(
                profile,
                condition_emb_gen,
                condition_emb_disc,
                batch_idx,
            )
        elif self.paradigm == "wgan":
            return self._training_step_wgan(
                profile,
                condition_emb_gen,
                condition_emb_disc,
                batch_idx,
            )
        else:
            raise ValueError(
                f'unrecognized GAN paradigm {self.paradigm}. \
                    need to be one of "vanilla" and "wgan"'
            )

    def _training_step_vanilla(
        self,
        profile,
        condition_emb_gen,
        condition_emb_disc,
        batch_idx,
    ):
        "training step of a vanilla GAN."
        batch_size = profile.shape[0]
        device = profile.device
        opt_g, opt_d = self.optimizers()
        condition_emb_disc_detached = condition_emb_disc.detach()

        # Train generator
        if batch_idx % self.d_steps_per_g == 0:
            self.toggle_optimizer(opt_g)
            opt_g.zero_grad()

            noise = torch.randn(batch_size, self.gan.latent_dim, device=device)
            fake_samples = self.gan.generator(noise, condition_emb_gen)
            fake_pred = self.gan.discriminator(
                fake_samples, condition_emb_disc_detached
            )
            g_loss = -fake_pred.mean()

            self.manual_backward(g_loss)
            self._clip_optimizer_gradients(opt_g)
            opt_g.step()
            self.untoggle_optimizer(opt_g)

            # log
            self.log(
                "Train/g_loss",
                g_loss.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        # Train discriminator
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()

        real_pred = self.gan.discriminator(profile, condition_emb_disc)
        _noise = torch.randn(batch_size, self.gan.latent_dim, device=device)
        with torch.no_grad():
            fake_samples = self.gan.generator(_noise, condition_emb_gen.detach())
        fake_pred = self.gan.discriminator(fake_samples, condition_emb_disc)
        # hinge loss
        d_real_loss = F.relu(1 - real_pred).mean()
        d_fake_loss = F.relu(1 + fake_pred).mean()
        d_loss = d_real_loss + d_fake_loss

        self.manual_backward(d_loss)
        self._clip_optimizer_gradients(opt_d)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.log(
            "Train/d_loss",
            d_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "Train/d_real_loss",
            d_real_loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Train/d_fake_loss",
            d_fake_loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return d_loss

    def _training_step_wgan(
        self,
        profile,
        condition_emb_gen,
        condition_emb_disc,
        batch_idx: int,
    ):
        opt_g, opt_d = self.optimizers()

        # Keep gradients for the discriminator embedder, but use a detached copy
        # whenever the generator queries the critic to avoid cross-updates.
        condition_emb_disc_detached = condition_emb_disc.detach()

        batch_size = profile.shape[0]
        device = profile.device

        # ===============================
        # Train Discriminator
        # ===============================
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad(set_to_none=True)

        # Real samples
        real_pred = self.gan.discriminator(profile, condition_emb_disc)
        d_real_loss = -torch.mean(real_pred)

        # Fake samples
        noise = torch.randn(batch_size, self.gan.latent_dim, device=device)
        with torch.no_grad():
            fake_samples = self.gan.generator(noise, condition_emb_gen.detach())
        fake_pred = self.gan.discriminator(fake_samples, condition_emb_disc)
        d_fake_loss = torch.mean(fake_pred)

        # Gradient penalty
        gp = self.gradient_penalty(profile, fake_samples, condition_emb_disc)
        d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gp

        self.manual_backward(d_loss)
        self._clip_optimizer_gradients(opt_d)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # ===============================
        # Train Generator (less frequently)
        # ===============================
        if batch_idx % self.d_steps_per_g == 0:  # configurable update frequency
            self.toggle_optimizer(opt_g)
            opt_g.zero_grad(set_to_none=True)

            # Generate fake samples
            noise = torch.randn(batch_size, self.gan.latent_dim, device=device)
            fake_samples = self.gan.generator(noise, condition_emb_gen)
            fake_pred = self.gan.discriminator(
                fake_samples, condition_emb_disc_detached
            )
            g_loss = -torch.mean(fake_pred)

            self.manual_backward(g_loss)
            self._clip_optimizer_gradients(opt_g)
            opt_g.step()
            self.untoggle_optimizer(opt_g)

            # Log generator loss
            self.log(
                "Train/g_loss",
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        # Log discriminator losses
        self.log(
            "Train/d_loss",
            d_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "Train/d_real_loss",
            d_real_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Train/d_fake_loss",
            d_fake_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("Train/gp", gp, on_step=True, on_epoch=True, sync_dist=True)

        return d_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        self.eval()
        profile, condition = batch

        # Get condition embedding
        condition_emb_gen = self._get_condition_embedding(
            condition, embedder=self.generator_label_embedder
        )
        disc_embedder = (
            self.discriminator_label_embedder or self.generator_label_embedder
        )
        condition_emb_disc = self._get_condition_embedding(
            condition, embedder=disc_embedder
        )

        # Generate samples for validation
        batch_size = profile.shape[0]
        device = profile.device
        noise = torch.randn(batch_size, self.gan.latent_dim, device=device)
        fake_samples = self.gan.generator(noise, condition_emb_gen)

        # Calculate validation losses
        real_pred = self.gan.discriminator(profile, condition_emb_disc)
        fake_pred = self.gan.discriminator(fake_samples, condition_emb_disc)

        d_real_loss = -torch.mean(real_pred)
        d_fake_loss = torch.mean(fake_pred)
        g_loss = -torch.mean(fake_pred)

        # Log validation losses
        self.log(
            "Validation/d_real_loss",
            d_real_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Validation/d_fake_loss",
            d_fake_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Validation/g_loss",
            g_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
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
        """Sample from the GAN given conditions.

        Args:
            condition: Dictionary of condition tensors
            num_samples: Number of samples to generate per condition
            return_latents: Whether to return the latent codes used

        Returns:
            Generated samples, optionally with latent codes. shape [bs_cond * num_samples]
        """
        self.eval()

        # Get condition embedding
        condition_emb = self._get_condition_embedding(
            condition, embedder=self.generator_label_embedder
        )

        batch_size = condition_emb.shape[0]
        device = condition_emb.device
        if return_latents:
            if num_samples > 1:
                z = torch.randn(
                    batch_size * num_samples, self.gan.latent_dim, device=device
                )
            else:
                z = torch.randn(batch_size, self.gan.latent_dim, device=device)
            samples = self.gan.generate(condition_emb, num_samples=num_samples, z=z)
            return samples, z
        else:
            samples = self.gan.generate(condition_emb, num_samples=num_samples)
            return samples
