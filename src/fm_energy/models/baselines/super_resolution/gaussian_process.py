"""Gaussian Process baseline for time series super-resolution."""

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tqdm import tqdm


class GaussianProcessBaseline:
    """Gaussian Process baseline for time series super-resolution."""

    def __init__(
        self,
        length_scale: float = 1.0,
        noise_level: float = 0.1,
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 5,
    ):
        """
        Initialize Gaussian Process baseline.

        Args:
            length_scale: Length scale parameter for RBF kernel
            noise_level: Noise level parameter for White kernel
            alpha: Value added to the diagonal of the kernel matrix during fitting
            n_restarts_optimizer: Number of restarts for hyperparameter optimization
        """
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer

        # Create kernel: RBF + White noise
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True,
        )

        self.is_fitted = False
        self.scale_factor = None  # Will be set during training

    def fit(self, time_series_list: list[torch.Tensor]):
        """
        Fit GP on training data.

        Args:
            time_series_list: List of high-resolution time series [B, T, D] or [B, T]
        """
        # Collect all training data
        all_x = []  # Time indices
        all_y = []  # Values

        for ts in time_series_list:
            if ts.dim() == 3:
                B, T, D = ts.shape
                ts_np = ts.cpu().numpy()
                for b in range(B):
                    for d in range(D):
                        x_indices = np.arange(T).reshape(-1, 1)  # [T, 1]
                        y_values = ts_np[b, :, d]  # [T]
                        all_x.append(x_indices)
                        all_y.append(y_values)
            else:
                B, T = ts.shape
                ts_np = ts.cpu().numpy()
                for b in range(B):
                    x_indices = np.arange(T).reshape(-1, 1)  # [T, 1]
                    y_values = ts_np[b, :]  # [T]
                    all_x.append(x_indices)
                    all_y.append(y_values)

        # Concatenate all data
        X_train = np.vstack(all_x)  # [N*T, 1]
        y_train = np.concatenate(all_y)  # [N*T]

        # Fit GP
        print("Fitting Gaussian Process...")
        self.gp.fit(X_train, y_train)
        self.is_fitted = True
        print("GP fitting completed!")

    def fit_dataloader(self, dataloader, scale_factor: int):
        """
        Fit GP using a dataloader that provides low-res/high-res pairs.

        Args:
            dataloader: DataLoader that yields (low_res_list, high_res_list) for variable-length models
            scale_factor: Scale factor for super-resolution
        """
        print("Training Gaussian Process...")
        self.scale_factor = scale_factor

        # Collect all training pairs
        all_x = []  # Time indices for low-res
        all_y = []  # Low-res values (what we observe)

        for low_res_list, _ in tqdm(dataloader, desc="Processing GP data"):
            for low_res in low_res_list:
                low_res = low_res.squeeze(0)  # Remove batch dim

                # Create time indices for low-res points (downsampled indices)
                low_res_indices = torch.arange(
                    0, len(low_res) * scale_factor, scale_factor, dtype=torch.float32
                )

                # Add data for training
                all_x.append(low_res_indices.unsqueeze(1))  # [T_low, 1]
                all_y.append(low_res.cpu().numpy())  # Low-res values

        # Concatenate all data
        X_train = np.vstack(all_x)  # [N*T, 1]
        y_train = np.concatenate(all_y)  # [N*T]

        # Fit GP
        print("Fitting Gaussian Process...")
        self.gp.fit(X_train, y_train)
        self.is_fitted = True
        print("GP fitting completed!")

    def super_resolve(
        self, time_series: torch.Tensor, scale_factor: int
    ) -> torch.Tensor:
        """
        Super-resolve time series using fitted GP.

        Args:
            time_series: Low-resolution time series [B, T] or [B, T, D]
            scale_factor: Upsampling factor

        Returns:
            Super-resolved time series [B, T*scale_factor] or [B, T*scale_factor, D]
        """
        if not self.is_fitted:
            raise RuntimeError("GP must be fitted before super-resolution")

        device = time_series.device
        original_shape = time_series.shape

        if time_series.dim() == 2:
            B, T = time_series.shape
            D = 1
            ts_np = time_series.cpu().numpy().reshape(B, T, 1)
        else:
            B, T, D = time_series.shape
            ts_np = time_series.cpu().numpy()

        # Create high-resolution time indices
        low_res_indices = np.arange(0, T, 1)  # [0, 1, 2, ..., T-1]
        high_res_indices = np.linspace(
            0, T - 1, T * scale_factor
        )  # Interpolate between

        super_resolved = np.zeros((B, T * scale_factor, D))

        for b in range(B):
            for d in range(D):
                # Low-resolution values
                y_low = ts_np[b, :, d]  # [T]

                # Create training data (low-res indices -> low-res values)
                X_low = low_res_indices.reshape(-1, 1)  # [T, 1]

                # Temporarily fit GP on this single series
                local_gp = GaussianProcessRegressor(
                    kernel=self.gp.kernel_, alpha=self.alpha, normalize_y=True
                )
                local_gp.fit(X_low, y_low)

                # Predict on high-resolution indices
                X_high = high_res_indices.reshape(-1, 1)  # [T*scale_factor, 1]
                y_high_pred, _ = local_gp.predict(X_high, return_std=True)

                super_resolved[b, :, d] = y_high_pred

        # Convert back to tensor
        super_resolved = torch.from_numpy(super_resolved).to(
            device=device, dtype=time_series.dtype
        )

        # Reshape to match input format
        if len(original_shape) == 2:
            super_resolved = super_resolved.squeeze(-1)  # [B, T*scale_factor]

        return super_resolved


class FastGaussianProcessBaseline:
    """Fast GP baseline using global kernel parameters with local fitting."""

    def __init__(
        self, length_scale: float = 1.0, noise_level: float = 0.1, alpha: float = 1e-10
    ):
        """
        Initialize Fast GP baseline.

        Args:
            length_scale: Fixed length scale for RBF kernel
            noise_level: Fixed noise level for White kernel
            alpha: Regularization parameter
        """
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.alpha = alpha
        self.is_fitted = True  # No global fitting needed

    def super_resolve(
        self, time_series: torch.Tensor, scale_factor: int
    ) -> torch.Tensor:
        """
        Super-resolve time series using GP with fixed hyperparameters.

        Args:
            time_series: Low-resolution time series [B, T] or [B, T, D]
            scale_factor: Upsampling factor

        Returns:
            Super-resolved time series [B, T*scale_factor] or [B, T*scale_factor, D]
        """
        device = time_series.device
        original_shape = time_series.shape

        if time_series.dim() == 2:
            B, T = time_series.shape
            D = 1
            ts_np = time_series.cpu().numpy().reshape(B, T, 1)
        else:
            B, T, D = time_series.shape
            ts_np = time_series.cpu().numpy()

        # Create kernel with fixed parameters
        kernel = RBF(length_scale=self.length_scale) + WhiteKernel(
            noise_level=self.noise_level
        )

        # Create high-resolution time indices
        low_res_indices = np.arange(0, T, 1)
        high_res_indices = np.linspace(0, T - 1, T * scale_factor)

        super_resolved = np.zeros((B, T * scale_factor, D))

        for b in range(B):
            for d in range(D):
                # Low-resolution values
                y_low = ts_np[b, :, d]

                # Fit GP on this single series with fixed kernel
                X_low = low_res_indices.reshape(-1, 1)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=self.alpha,
                    normalize_y=True,
                    optimizer=None,  # Don't optimize hyperparameters
                )
                gp.fit(X_low, y_low)

                # Predict on high-resolution indices
                X_high = high_res_indices.reshape(-1, 1)
                y_high_pred = gp.predict(X_high)

                super_resolved[b, :, d] = y_high_pred

        # Convert back to tensor
        super_resolved = torch.from_numpy(super_resolved).to(
            device=device, dtype=time_series.dtype
        )

        # Reshape to match input format
        if len(original_shape) == 2:
            super_resolved = super_resolved.squeeze(-1)

        return super_resolved
