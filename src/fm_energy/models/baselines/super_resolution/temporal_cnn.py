"""Temporal CNN baseline for time series super-resolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class TemporalCNNModel(nn.Module):
    """Temporal CNN model for time series super-resolution."""

    def __init__(
        self,
        input_dim: int,
        scale_factor: int,
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.scale_factor = scale_factor
        self.hidden_channels = hidden_channels

        # Initial feature extraction
        self.input_conv = nn.Conv1d(
            input_dim, hidden_channels, kernel_size=7, padding=3
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels, kernel_size) for _ in range(num_layers)]
        )

        # Upsampling layers
        self.upsampling = nn.ModuleList()
        current_scale = 1
        current_channels = hidden_channels

        while current_scale < scale_factor:
            if current_scale * 2 <= scale_factor:
                # Use transposed conv for 2x upsampling
                self.upsampling.append(
                    nn.ConvTranspose1d(
                        current_channels,
                        current_channels // 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
                current_scale *= 2
                current_channels = current_channels // 2
            else:
                # Use interpolation + conv for non-power-of-2 factors
                remaining_factor = scale_factor // current_scale
                self.upsampling.append(
                    nn.Sequential(
                        nn.Upsample(
                            scale_factor=remaining_factor,
                            mode="linear",
                            align_corners=False,
                        ),
                        nn.Conv1d(
                            current_channels, current_channels, kernel_size=3, padding=1
                        ),
                    )
                )
                current_scale = scale_factor

        # Final output layer
        self.output_conv = nn.Conv1d(
            current_channels, input_dim, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, D, T]

        Returns:
            Super-resolved tensor [B, D, T*scale_factor]
        """
        # Initial feature extraction
        out = F.relu(self.input_conv(x))

        # Residual blocks
        for res_block in self.res_blocks:
            out = res_block(out)

        # Upsampling
        for layer in self.upsampling:
            out = F.relu(layer(out))

        # Final output
        out = self.output_conv(out)

        return out


class ResidualBlock(nn.Module):
    """Residual block for temporal CNN."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class TemporalCNNBaseline:
    """Temporal CNN baseline for time series super-resolution."""

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        device: str | None = None,
    ):
        """
        Initialize Temporal CNN baseline.

        Args:
            hidden_channels: Number of hidden channels
            num_layers: Number of residual blocks
            kernel_size: Kernel size for convolutions
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to use
        """
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.models = {}  # Store different models for different scale factors
        self.is_fitted = {}

    def _create_training_pairs(
        self, time_series_list: list[torch.Tensor], scale_factor: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create low-res/high-res training pairs by downsampling.

        Args:
            time_series_list: List of high-resolution time series
            scale_factor: Scale factor for downsampling

        Returns:
            Tuple of (low_res_data, high_res_data)
        """
        low_res_data = []
        high_res_data = []

        for ts in time_series_list:
            if ts.dim() == 3:
                _B, _T, _D = ts.shape
                # Transpose to [B, D, T] for conv1d
                ts_transposed = ts.transpose(1, 2)
            else:
                _B, _T = ts.shape
                _D = 1
                ts_transposed = ts.unsqueeze(1)  # [B, 1, T]

            # Downsample to create low-resolution version
            low_res = F.avg_pool1d(
                ts_transposed, kernel_size=scale_factor, stride=scale_factor
            )

            # Ensure high-res version is compatible length
            target_length = low_res.shape[-1] * scale_factor
            if ts_transposed.shape[-1] > target_length:
                high_res = ts_transposed[:, :, :target_length]
            else:
                high_res = ts_transposed

            low_res_data.append(low_res)
            high_res_data.append(high_res)

        return torch.cat(low_res_data, dim=0), torch.cat(high_res_data, dim=0)

    def fit(self, dataloader, scale_factor: int):
        """
        Fit Temporal CNN on training data using DataLoader with pre-processed low-res/high-res pairs.

        Args:
            dataloader: DataLoader that yields (low_res_batch, high_res_batch)
            scale_factor: Scale factor for super-resolution
        """
        print(f"Training Temporal CNN for scale factor {scale_factor}...")

        # Determine input dimension from first batch
        first_batch = next(iter(dataloader))
        low_res_batch, _ = first_batch
        input_dim = low_res_batch.shape[-1] if low_res_batch.dim() == 3 else 1

        # Initialize model for this scale factor
        self.models[scale_factor] = TemporalCNNModel(
            input_dim=input_dim,
            scale_factor=scale_factor,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
        ).to(self.device)

        model = self.models[scale_factor]

        # Print model parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"TemporalCNN model (scale={scale_factor}) initialized with {num_params:,} trainable parameters"
        )
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop - use the dataloader directly
        model.train()
        for epoch in tqdm(range(self.epochs), desc="CNN Training"):
            total_loss = 0
            num_batches = 0

            for batch_low, batch_high in tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False
            ):
                batch_low = batch_low.to(self.device)
                batch_high = batch_high.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                pred_high = model(batch_low)

                # Compute loss
                loss = criterion(pred_high, batch_high)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / num_batches
                print(
                    f"TemporalCNN (scale={scale_factor}) Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}"
                )

        self.is_fitted[scale_factor] = True

    def fit_legacy(self, time_series_list: list[torch.Tensor], scale_factor: int):
        """Legacy fit method for backwards compatibility."""
        # Create training pairs using the old method
        low_res_data, high_res_data = self._create_training_pairs(
            time_series_list, scale_factor
        )

        # Create DataLoader and use new fit method
        dataset = torch.utils.data.TensorDataset(low_res_data, high_res_data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.fit(dataloader, scale_factor)

    def fit_dataloader(self, dataloader, scale_factor: int):
        """
        Fit Temporal CNN using a dataloader.

        Args:
            dataloader: DataLoader that yields (low_res_batch, high_res_batch)
            scale_factor: Scale factor for super-resolution
        """
        print("Training Temporal CNN...")
        self.fit(dataloader, scale_factor)

    def super_resolve(
        self, time_series: torch.Tensor, scale_factor: int
    ) -> torch.Tensor:
        """
        Super-resolve time series using trained Temporal CNN.

        Args:
            time_series: Low-resolution time series [B, T] or [B, T, D]
            scale_factor: Scale factor for super-resolution

        Returns:
            Super-resolved time series [B, T*scale_factor] or [B, T*scale_factor, D]
        """
        if scale_factor not in self.is_fitted or not self.is_fitted[scale_factor]:
            raise RuntimeError(
                f"TemporalCNN must be fitted for scale_factor={scale_factor} before super-resolution"
            )

        model = self.models[scale_factor]
        model.eval()

        with torch.no_grad():
            _original_shape = time_series.shape
            device = time_series.device

            if time_series.dim() == 2:
                B, T = time_series.shape
                # Add channel dimension: [B, T] -> [B, 1, T]
                input_tensor = time_series.unsqueeze(1).to(self.device)
                squeeze_channel = True
            else:
                B, T, D = time_series.shape
                # Transpose: [B, T, D] -> [B, D, T]
                input_tensor = time_series.transpose(1, 2).to(self.device)
                squeeze_channel = False

            # Super-resolve
            super_resolved = model(input_tensor)

            # Reshape back to original format
            if squeeze_channel:
                super_resolved = super_resolved.squeeze(1)  # [B, T*scale_factor]
            else:
                super_resolved = super_resolved.transpose(
                    1, 2
                )  # [B, T*scale_factor, D]

        return super_resolved.to(device)
