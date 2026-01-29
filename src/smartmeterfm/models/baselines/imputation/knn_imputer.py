"""KNN-based imputation baseline using scikit-learn."""

import numpy as np
import torch
from sklearn.impute import KNNImputer
from tqdm import tqdm


class KNNImputerBaseline:
    """KNN-based imputation using scikit-learn's KNNImputer."""

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        """
        Initialize KNN imputer.

        Args:
            n_neighbors: Number of neighboring samples to use for imputation
            weights: Weight function used in prediction ('uniform' or 'distance')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        self.is_fitted = False

    def fit(
        self,
        time_series_list: list[torch.Tensor],
        masks_list: list[torch.Tensor] | None = None,
    ):
        """
        Fit the KNN imputer on training data.

        Args:
            time_series_list: List of time series tensors [B, T] or [B, T, D]
            masks_list: Optional list of masks (if None, assume no missing values in training)
        """
        # Concatenate all time series for training
        all_data = []

        for _i, ts in enumerate(time_series_list):
            if ts.dim() == 3:
                # [B, T, D] -> reshape to [B*D, T]
                B, T, D = ts.shape
                ts_reshaped = ts.transpose(1, 2).reshape(B * D, T)
            else:
                # [B, T] -> [B, T]
                ts_reshaped = ts

            all_data.append(ts_reshaped)

        # Concatenate along batch dimension
        training_data = torch.cat(all_data, dim=0).cpu().numpy()  # [N, T]

        # Apply masks if provided
        if masks_list is not None:
            all_masks = []
            for _i, (ts, mask) in enumerate(
                zip(time_series_list, masks_list, strict=False)
            ):
                if ts.dim() == 3:
                    B, T, D = ts.shape
                    mask_reshaped = mask.transpose(1, 2).reshape(B * D, T)
                else:
                    mask_reshaped = mask
                all_masks.append(mask_reshaped)

            training_masks = torch.cat(all_masks, dim=0).cpu().numpy()  # [N, T]
            # Set missing values to NaN
            training_data[training_masks == 0] = np.nan

        # Fit the imputer
        self.imputer.fit(training_data)
        self.is_fitted = True

    def fit_dataloader(self, dataloader, max_train_samples: int = None):
        """
        Fit KNN imputer using a dataloader.

        Args:
            dataloader: DataLoader that yields (ts_batch, mask_batch) for fixed-length models
        """
        print("Collecting training data for KNN...")
        all_data = []
        all_masks = []

        for ts_batch, mask_batch in tqdm(dataloader, desc="Collecting KNN data"):
            # Convert batched tensors to list format expected by fit()
            batch_size = ts_batch.shape[0]
            for i in range(batch_size):
                all_data.append(ts_batch[i].unsqueeze(0))  # [1, T]
                all_masks.append(mask_batch[i].unsqueeze(0))  # [1, T]
        if max_train_samples is not None:
            all_data = all_data[:max_train_samples]
            all_masks = all_masks[:max_train_samples]

        print(f"Training KNN on {len(all_data)} samples...")
        self.fit(all_data, all_masks)

    def impute(self, time_series: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Impute missing values in time series.

        Args:
            time_series: Input time series [B, T] or [B, T, D]
            mask: Boolean mask where 1=observed, 0=missing

        Returns:
            Imputed time series with same shape as input
        """
        if not self.is_fitted:
            raise RuntimeError("KNNImputer must be fitted before imputation")

        original_shape = time_series.shape
        device = time_series.device

        if time_series.dim() == 3:
            B, T, D = time_series.shape
            # Reshape to [B*D, T]
            ts_reshaped = time_series.transpose(1, 2).reshape(B * D, T)
            mask_reshaped = mask.transpose(1, 2).reshape(B * D, T)
        else:
            ts_reshaped = time_series
            mask_reshaped = mask

        # Convert to numpy and set missing values to NaN
        data_np = ts_reshaped.cpu().numpy()
        mask_np = mask_reshaped.cpu().numpy()
        data_np[mask_np == 0] = np.nan

        # Perform imputation
        imputed_np = self.imputer.transform(data_np)

        # Convert back to tensor
        imputed = torch.from_numpy(imputed_np).to(
            device=device, dtype=time_series.dtype
        )

        # Reshape back to original shape
        if len(original_shape) == 3:
            B, T, D = original_shape
            imputed = imputed.reshape(B, D, T).transpose(1, 2)

        return imputed
