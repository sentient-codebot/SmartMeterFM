"""ProfileSR: A GAN-based Profile Super-Resolution Model

This is a GAN-based baseline for super-resolution that upsamples low-resolution
profiles to high-resolution profiles with scale factors of 2, 4, 8, or 16.
"""

import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from torch import nn
from torch.nn.utils import spectral_norm
from tqdm import tqdm


class ResidualBlock(nn.Module):
    """Residual block with 1D convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ProfileSRGenerator(nn.Module):
    """Generator for ProfileSR that upsamples low-resolution profiles."""

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

        # Initial convolution
        self.initial_conv = nn.Conv1d(1, 128, kernel_size=7, stride=1, padding=3)
        self.initial_bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=False)

        # Residual blocks (retain feature map size)
        self.residual_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(4)])

        # Upsampling layers using transposed convolution
        self.upsample_layers = nn.ModuleList()
        current_channels = 128

        # Calculate number of upsampling layers needed
        num_upsample = int(scale_factor).bit_length() - 1  # log2(scale_factor)

        for i in range(num_upsample):
            next_channels = current_channels // 2 if i > 0 else current_channels
            self.upsample_layers.append(
                nn.ConvTranspose1d(
                    current_channels, next_channels, kernel_size=4, stride=2, padding=1
                )
            )
            self.upsample_layers.append(nn.BatchNorm1d(next_channels))
            self.upsample_layers.append(nn.ReLU(inplace=False))
            current_channels = next_channels

        # Final output layer
        self.final_conv = nn.Conv1d(
            current_channels, 1, kernel_size=7, stride=1, padding=3
        )

    def forward(self, x):
        """
        Forward pass of generator.

        Args:
            x: Input low-resolution profile [B, T, 1]

        Returns:
            High-resolution profile [B, T * scale_factor, 1]
        """
        # Change to conv shape (B, C, 1)
        x = x.permute(0, 2, 1)

        # Initial convolution
        x = self.initial_conv(x)  # [B, d, T]
        x = self.initial_bn(x)  # [B, d, T]
        x = self.relu(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)  # [B, d, T]

        # Upsampling
        for layer in self.upsample_layers:
            x = layer(x)  # final: [B, < d, T * scale_factor]

        # Final output
        x = self.final_conv(x)  # [B, 1, T * scale_factor]

        # Change back to (B, T, C)
        x = x.permute(0, 2, 1)

        return x


class ProfileSRDiscriminator(nn.Module):
    """Discriminator for ProfileSR that classifies high-resolution profiles."""

    def __init__(self, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length

        # Convolutional layers with spectral normalization
        self.conv_layers = nn.ModuleList(
            [
                spectral_norm(nn.Conv1d(1, 8, kernel_size=4, stride=2, padding=1)),
                spectral_norm(nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=1)),
                spectral_norm(nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1)),
                spectral_norm(nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)),
            ]
        )

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)

        # Calculate final feature size
        final_size = self._calculate_final_size()

        # Final classification layer
        self.final_conv = spectral_norm(nn.Linear(final_size, 1))

    def _calculate_final_size(self):
        """Calculate the size after all conv layers."""
        size = self.sequence_length
        for _ in self.conv_layers:
            size = (size - 1) // 2 + 1  # stride=2, kernel=4, padding=1
        return size * 64  # 64 is the final number of channels

    def forward(self, x):
        """
        Forward pass of discriminator.

        Args:
            x: High-resolution profile [B, T, 1]

        Returns:
            tuple: (realness [B, 1], features [list of tensors])
        """
        # Change to conv shape (B, C, T)
        x = x.permute(0, 2, 1)

        features = []
        for conv in self.conv_layers:
            x = conv(x)
            x = self.leaky_relu(x)
            features.append(x)

        # Flatten and classify
        x = x.flatten(start_dim=1)
        realness = self.final_conv(x)

        return realness, features


class ProfileSRPolish(nn.Module):
    """Polish network that refines the generator output."""

    def __init__(self):
        super().__init__()

        # Residual blocks with same input/output size
        self.residual_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(3)])

        # Input/output convolutions
        self.input_conv = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)
        self.input_bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=False)

        self.output_conv = nn.Conv1d(64, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        """
        Forward pass of polish network.

        Args:
            x: Raw high-resolution profile [B, T, 1]

        Returns:
            Polished high-resolution profile [B, T, 1]
        """
        # Change to conv shape (B, C, T)
        x = x.permute(0, 2, 1)

        # Input processing
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.relu(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output
        x = self.output_conv(x)

        # Change back to (B, T, C)
        x = x.permute(0, 2, 1)

        return x


class ProfileSRBaseline:
    """ProfileSR baseline model for super-resolution tasks."""

    def __init__(
        self,
        scale_factor: int = 2,
        sequence_length: int = None,  # Will be calculated based on scale_factor
        learning_rate: float = 0.0002,
        stage1_epochs: int = 100,
        stage2_epochs: int = 50,
        batch_size: int = 32,
        device: str = None,
        use_wandb: bool = False,
        wandb_project: str = "fm-energy-super-resolution",
        wandb_run_name: str = None,
    ):
        """
        Args:
            scale_factor: Upsampling factor (2, 4, 8, or 16)
            sequence_length: Length of high-resolution sequence (T * scale_factor)
            learning_rate: Learning rate for optimizers
            stage1_epochs: Epochs for training generator and discriminator
            stage2_epochs: Epochs for training polish network
            batch_size: Batch size for training
            device: Device to use for training
            use_wandb: Whether to use wandb logging
            wandb_project: Wandb project name
            wandb_run_name: Wandb run name (optional)
        """
        if scale_factor not in [2, 4, 8, 16]:
            raise ValueError("scale_factor must be one of [2, 4, 8, 16]")

        self.scale_factor = scale_factor
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.batch_size = batch_size
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize networks
        self.generator = ProfileSRGenerator(scale_factor).to(self.device)
        self.discriminator = ProfileSRDiscriminator(sequence_length).to(self.device)
        self.polish = ProfileSRPolish().to(self.device)

        # Initialize optimizers
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
        self.optimizer_p = torch.optim.Adam(
            self.polish.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )

        self.is_stage1_fitted = False
        self.is_stage2_fitted = False

    def _generator_loss(self, fake_hr, real_hr):
        """Calculate generator losses."""
        # Content loss (MSE)
        content_loss = F.mse_loss(fake_hr, real_hr)

        # Adversarial loss
        fake_realness, fake_features = self.discriminator(fake_hr)
        adversarial_loss = -fake_realness.mean()

        # Feature matching loss
        real_realness, real_features = self.discriminator(real_hr)
        feature_loss = 0.0
        for fake_feat, real_feat in zip(fake_features, real_features, strict=True):
            feature_loss += F.mse_loss(fake_feat, real_feat)
        feature_loss /= len(fake_features)

        # Combined loss with weights from description
        total_loss = content_loss + 0.05 * adversarial_loss + 0.5 * feature_loss

        return total_loss, content_loss, adversarial_loss, feature_loss

    def _discriminator_loss(self, fake_hr, real_hr):
        """Calculate discriminator loss."""
        # Real samples
        real_realness, _ = self.discriminator(real_hr)

        # Fake samples (detached to prevent generator gradients)
        fake_realness, _ = self.discriminator(fake_hr.detach())

        # Hinge loss
        return (F.relu(1 - real_realness) + F.relu(1 + fake_realness)).mean()

    def _polish_loss(self, polished_hr, real_hr):
        """Calculate polish network losses."""

        # Outline loss with max pooling
        def outline_loss(pred, target):
            # Max pooling with kernel_size=3, stride=1
            pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

            # Apply max pooling
            pred_pooled = rearrange(
                pool(rearrange(pred, "B T C -> B C T")), "B C T -> B T C"
            )
            target_pooled = rearrange(
                pool(rearrange(target, "B T C -> B C T")), "B C T -> B T C"
            )
            loss1 = F.mse_loss(pred_pooled, target_pooled)

            # Reverse sign and apply again
            pred_neg_pooled = rearrange(
                pool(rearrange((-pred), "B T C -> B C T")), "B C T -> B T C"
            )
            target_neg_pooled = rearrange(
                pool(rearrange((-target), "B T C -> B C T")), "B C T -> B T C"
            )
            loss2 = F.mse_loss(pred_neg_pooled, target_neg_pooled)

            return (loss1 + loss2) / 2

        # Switch loss with first-order difference and max pooling
        def switch_loss(pred, target):
            # First-order difference on absolute values
            pred_abs = torch.abs(pred)
            target_abs = torch.abs(target)

            pred_diff = pred_abs[:, 1:] - pred_abs[:, :-1]
            target_diff = target_abs[:, 1:] - target_abs[:, :-1]

            # Add dimension for pooling
            pred_diff = pred_diff.unsqueeze(-1) if pred_diff.dim() == 2 else pred_diff
            target_diff = (
                target_diff.unsqueeze(-1) if target_diff.dim() == 2 else target_diff
            )

            # Max pooling
            pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            pred_pooled = rearrange(
                pool(rearrange(pred_diff, "B T C -> B C T")), "B C T -> B T C"
            )
            target_pooled = rearrange(
                pool(rearrange(target_diff, "B T C -> B C T")), "B C T -> B T C"
            )

            return F.mse_loss(pred_pooled, target_pooled)

        outline_l = outline_loss(polished_hr, real_hr)
        switch_l = switch_loss(polished_hr, real_hr)

        return outline_l + switch_l, outline_l, switch_l

    def _initialize_wandb(self, stage: str):
        """Initialize wandb if enabled."""
        if self.use_wandb:
            run_name = (
                f"{self.wandb_run_name}_{stage}"
                if self.wandb_run_name
                else f"ProfileSR_{stage}"
            )
            wandb.init(
                project=self.wandb_project,
                name=run_name,
                config={
                    "scale_factor": self.scale_factor,
                    "sequence_length": self.sequence_length,
                    "learning_rate": self.learning_rate,
                    "stage1_epochs": self.stage1_epochs,
                    "stage2_epochs": self.stage2_epochs,
                    "batch_size": self.batch_size,
                    "device": str(self.device),
                    "stage": stage,
                },
            )

    def fit_stage1(self, dataloader):
        """
        Stage 1: Train generator and discriminator.

        Args:
            dataloader: DataLoader that yields (low_res_batch, high_res_batch)
        """
        self._initialize_wandb("stage1")

        print("Training ProfileSR Stage 1 (Generator + Discriminator)...")

        # Print model parameters
        gen_params = sum(
            p.numel() for p in self.generator.parameters() if p.requires_grad
        )
        disc_params = sum(
            p.numel() for p in self.discriminator.parameters() if p.requires_grad
        )
        total_params = gen_params + disc_params

        print(f"Stage 1 initialized with {total_params:,} trainable parameters")
        print(f"  - Generator: {gen_params:,}")
        print(f"  - Discriminator: {disc_params:,}")

        if self.use_wandb:
            wandb.log(
                {
                    "model_parameters_total": total_params,
                    "model_parameters_generator": gen_params,
                    "model_parameters_discriminator": disc_params,
                }
            )

        # Training loop
        pbar_epoch = tqdm(range(self.stage1_epochs), desc="Stage 1 Training")
        for epoch in pbar_epoch:
            total_gen_loss = 0
            total_disc_loss = 0
            num_batches = 0

            pbar_batch = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.stage1_epochs}", leave=False
            )
            for low_res_batch, high_res_batch in pbar_batch:
                low_res_batch = low_res_batch.to(self.device)
                high_res_batch = high_res_batch.to(self.device)

                # Train Generator
                self.optimizer_g.zero_grad()
                fake_hr = self.generator(low_res_batch)
                gen_loss, content_loss, adv_loss, feat_loss = self._generator_loss(
                    fake_hr, high_res_batch
                )
                gen_loss.backward()
                self.optimizer_g.step()

                # Train Discriminator
                self.optimizer_d.zero_grad()
                disc_loss = self._discriminator_loss(fake_hr, high_res_batch)
                disc_loss.backward()
                self.optimizer_d.step()

                total_gen_loss += gen_loss.item()
                total_disc_loss += disc_loss.item()

                # Log batch losses to wandb
                if self.use_wandb:
                    wandb.log(
                        {
                            "stage1/batch_gen_loss": gen_loss.item(),
                            "stage1/batch_content_loss": content_loss.item(),
                            "stage1/batch_adv_loss": adv_loss.item(),
                            "stage1/batch_feat_loss": feat_loss.item(),
                            "stage1/batch_disc_loss": disc_loss.item(),
                        }
                    )

                pbar_batch.set_postfix(
                    {
                        "G": f"{gen_loss.item():.3f}",
                        "D": f"{disc_loss.item():.3f}",
                    }
                )

                num_batches += 1

            # Average losses for the epoch
            avg_gen_loss = total_gen_loss / num_batches
            avg_disc_loss = total_disc_loss / num_batches

            pbar_epoch.set_postfix(
                {
                    "GenL": f"{avg_gen_loss:.3f}",
                    "DiscL": f"{avg_disc_loss:.3f}",
                }
            )

            # Log to wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "stage1/epoch_gen_loss": avg_gen_loss,
                        "stage1/epoch_disc_loss": avg_disc_loss,
                        "stage1/epoch": epoch + 1,
                    }
                )

            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Stage 1 Epoch {epoch + 1}/{self.stage1_epochs}")
                print(f"  Generator Loss: {avg_gen_loss:.4f}")
                print(f"  Discriminator Loss: {avg_disc_loss:.4f}")

        self.is_stage1_fitted = True
        print("Stage 1 training completed.")

        if self.use_wandb:
            wandb.finish()

    def fit_stage2(self, dataloader):
        """
        Stage 2: Freeze generator and train polish network.

        Args:
            dataloader: DataLoader that yields (low_res_batch, high_res_batch)
        """
        if not self.is_stage1_fitted:
            raise RuntimeError("Stage 1 must be completed before Stage 2")

        self._initialize_wandb("stage2")

        print("Training ProfileSR Stage 2 (Polish Network)...")

        # Freeze generator
        for param in self.generator.parameters():
            param.requires_grad = False

        polish_params = sum(
            p.numel() for p in self.polish.parameters() if p.requires_grad
        )
        print(
            f"Stage 2 initialized with {polish_params:,} trainable parameters (Polish Network)"
        )

        if self.use_wandb:
            wandb.log({"model_parameters_polish": polish_params})

        # Training loop
        pbar_epoch = tqdm(range(self.stage2_epochs), desc="Stage 2 Training")
        for epoch in pbar_epoch:
            total_polish_loss = 0
            num_batches = 0

            pbar_batch = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.stage2_epochs}", leave=False
            )
            for low_res_batch, high_res_batch in pbar_batch:
                low_res_batch = low_res_batch.to(self.device)
                high_res_batch = high_res_batch.to(self.device)

                # Generate raw high-res with frozen generator
                with torch.no_grad():
                    raw_hr = self.generator(low_res_batch)

                # Train Polish Network
                self.optimizer_p.zero_grad()
                polished_hr = self.polish(raw_hr)
                polish_loss, outline_loss, switch_loss = self._polish_loss(
                    polished_hr, high_res_batch
                )
                polish_loss.backward()
                self.optimizer_p.step()

                total_polish_loss += polish_loss.item()

                # Log batch losses to wandb
                if self.use_wandb:
                    wandb.log(
                        {
                            "stage2/batch_polish_loss": polish_loss.item(),
                            "stage2/batch_outline_loss": outline_loss.item(),
                            "stage2/batch_switch_loss": switch_loss.item(),
                        }
                    )

                pbar_batch.set_postfix(
                    {
                        "P": f"{polish_loss.item():.3f}",
                    }
                )

                num_batches += 1

            # Average loss for the epoch
            avg_polish_loss = total_polish_loss / num_batches

            pbar_epoch.set_postfix(
                {
                    "PolishL": f"{avg_polish_loss:.3f}",
                }
            )

            # Log to wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "stage2/epoch_polish_loss": avg_polish_loss,
                        "stage2/epoch": epoch + 1,
                    }
                )

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Stage 2 Epoch {epoch + 1}/{self.stage2_epochs}")
                print(f"  Polish Loss: {avg_polish_loss:.4f}")

        self.is_stage2_fitted = True
        print("Stage 2 training completed.")

        if self.use_wandb:
            wandb.finish()

    def fit(self, dataloader):
        """
        Complete training: Stage 1 followed by Stage 2.

        Args:
            dataloader: DataLoader that yields (low_res_batch, high_res_batch)
        """
        self.fit_stage1(dataloader)
        self.fit_stage2(dataloader)

    def super_resolve(self, low_res_profiles):
        """
        Super-resolve low-resolution profiles to high-resolution.

        Args:
            low_res_profiles: Low-resolution profiles [B, T, 1]

        Returns:
            High-resolution profiles [B, T * scale_factor, 1]
        """
        if not self.is_stage1_fitted:
            raise RuntimeError("Stage 1 must be completed before super-resolution")

        # Set models to evaluation mode
        self.generator.eval()
        if self.is_stage2_fitted:
            self.polish.eval()

        with torch.no_grad():
            low_res_profiles = low_res_profiles.to(self.device)

            # Generate high-resolution with generator
            high_res = self.generator(low_res_profiles)

            # Apply polish if available
            if self.is_stage2_fitted:
                high_res = self.polish(high_res)

        return high_res.cpu()
