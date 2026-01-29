"""Training-free interpolation baselines for imputation and super-resolution."""

from typing import Literal

import torch
import torch.nn.functional as F


class InterpolationBaseline:
    """Training-free baseline using interpolation methods."""

    def __init__(self, method: Literal["nearest", "linear"] = "linear"):
        """
        Initialize interpolation baseline.

        Args:
            method: Interpolation method ("nearest" or "linear")
        """
        self.method = method

    def impute(self, time_series: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Impute missing values in time series using interpolation.

        Args:
            time_series: Input time series [T] or [B, T] or [B, T, D]
            mask: Boolean mask where 1=observed, 0=missing [T] or [B, T] or [B, T, D]

        Returns:
            Imputed time series with same shape as input
        """
        if time_series.dim() == 1:
            time_series = time_series.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            mask = mask.unsqueeze(0).unsqueeze(-1)
            squeeze_output = True
        elif time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)  # [B, T, 1]
            mask = mask.unsqueeze(-1)
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, D = time_series.shape
        imputed = time_series.clone()

        for b in range(B):
            for d in range(D):
                ts = time_series[b, :, d]  # [T]
                m = mask[b, :, d]  # [T]

                # Get observed indices and values
                observed_indices = torch.where(m == 1)[0]
                if len(observed_indices) == 0:
                    continue  # Skip if no observed values

                observed_values = ts[observed_indices]

                # Get missing indices
                missing_indices = torch.where(m == 0)[0]
                if len(missing_indices) == 0:
                    continue  # Skip if no missing values

                # Interpolate missing values
                if self.method == "nearest":
                    # Find nearest observed value for each missing index
                    for missing_idx in missing_indices:
                        distances = torch.abs(
                            observed_indices.float() - missing_idx.float()
                        )
                        nearest_idx = torch.argmin(distances)
                        imputed[b, missing_idx, d] = observed_values[nearest_idx]

                elif self.method == "linear":
                    # Linear interpolation
                    for missing_idx in missing_indices:
                        # Find surrounding observed points
                        left_indices = observed_indices[observed_indices < missing_idx]
                        right_indices = observed_indices[observed_indices > missing_idx]

                        if len(left_indices) == 0:
                            # Extrapolate from right
                            imputed[b, missing_idx, d] = observed_values[0]
                        elif len(right_indices) == 0:
                            # Extrapolate from left
                            imputed[b, missing_idx, d] = observed_values[-1]
                        else:
                            # Interpolate between left and right
                            left_idx = left_indices[-1]  # Closest left point
                            right_idx = right_indices[0]  # Closest right point

                            left_val = ts[left_idx]
                            right_val = ts[right_idx]

                            # Linear interpolation
                            alpha = (missing_idx - left_idx).float() / (
                                right_idx - left_idx
                            ).float()
                            imputed[b, missing_idx, d] = (
                                1 - alpha
                            ) * left_val + alpha * right_val

        if squeeze_output:
            if time_series.shape[-1] == 1:
                imputed = imputed.squeeze(-1)
            if imputed.shape[0] == 1:
                imputed = imputed.squeeze(0)

        return imputed

    def super_resolve(
        self, time_series: torch.Tensor, scale_factor: int
    ) -> torch.Tensor:
        """
        Super-resolve time series by upsampling with interpolation.

        Args:
            time_series: Input low-resolution time series [B, T] or [B, T, D]
            scale_factor: Upsampling factor (2, 4, etc.)

        Returns:
            Super-resolved time series [B, T*scale_factor] or [B, T*scale_factor, D]
        """
        if time_series.dim() == 1:
            time_series = time_series.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        if time_series.dim() == 2:
            # [B, T] -> [B, 1, T] for interpolate
            time_series = time_series.unsqueeze(1)
            squeeze_channel = True
        else:
            # [B, T, D] -> [B, D, T] for interpolate
            time_series = time_series.transpose(1, 2)
            squeeze_channel = False

        # Use PyTorch's interpolate function
        mode = "nearest" if self.method == "nearest" else "linear"
        upsampled = F.interpolate(
            time_series,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False if mode == "linear" else None,
        )

        if squeeze_channel:
            upsampled = upsampled.squeeze(1)  # [B, T*scale_factor]
        else:
            upsampled = upsampled.transpose(1, 2)  # [B, T*scale_factor, D]

        if squeeze_output:
            upsampled = upsampled.squeeze(0)

        return upsampled
