"""Wrapper dataloader for baseline model training."""

from typing import Literal

import torch


def generate_mcar_mask(
    length: int, missing_rate: float, seed: int = None
) -> torch.Tensor:
    """Generate Missing Completely At Random (MCAR) mask for training."""
    if seed is not None:
        torch.manual_seed(seed)
    mask = torch.ones(length)
    num_missing = int(length * missing_rate)
    missing_indices = torch.randperm(length)[:num_missing]
    mask[missing_indices] = 0.0
    return mask


def generate_mnar_consecutive_mask(
    length: int, missing_rate: float, min_block_size: int = 5, seed: int = None
) -> torch.Tensor:
    """Generate Missing Not At Random (MNAR) mask with consecutive block pattern.

    Returns:
        torch.Tensor: Mask tensor where 1=observed, 0=missing. shape [length].

    """
    if seed is not None:
        torch.manual_seed(seed)
    mask = torch.ones(length)
    target_missing = int(length * missing_rate)
    current_missing = 0
    max_attempts = 100
    attempts = 0

    while current_missing < target_missing and attempts < max_attempts:
        remaining_needed = target_missing - current_missing
        max_block_size = min(remaining_needed, length // 4)
        block_size = torch.randint(
            min_block_size, max(min_block_size + 1, max_block_size + 1), (1,)
        ).item()
        max_start = length - block_size
        if max_start <= 0:
            break
        start_pos = torch.randint(0, max_start + 1, (1,)).item()
        end_pos = start_pos + block_size
        mask[start_pos:end_pos] = 0.0
        current_missing += block_size
        attempts += 1
    return mask


class BaselineDataLoader:
    """Wrapper dataloader that processes data for baseline training."""

    def __init__(
        self,
        original_dataloader,
        task_type="imputation",
        missing_rate=0.30,
        scale_factor=None,
        imputation_type: Literal["mcar", "mnar_consecutive"] = "mcar",
    ):
        """
        Initialize baseline dataloader wrapper.

        Args:
            original_dataloader: Original streaming dataloader
            task_type: "imputation" or "super_resolution"
            missing_rate: Missing rate for imputation training masks
            scale_factor: Scale factor for super-resolution downsampling (required for SR)
        """
        self.original_dataloader = original_dataloader
        self.task_type = task_type
        self.missing_rate = missing_rate
        self.scale_factor = scale_factor
        self._global_sample_idx = 0
        self.imputation_type = imputation_type

        if task_type == "super_resolution" and scale_factor is None:
            raise ValueError("scale_factor is required for super_resolution task")

    def __iter__(self):
        self._global_sample_idx = 0
        for profiles, labels in self.original_dataloader:
            batch_size = profiles.shape[0]

            batch_target_data = []
            batch_masks = []
            batch_downsampled = []
            batch_validity = []

            for i in range(batch_size):
                profile = profiles[i]
                sample_labels = {key: labels[key][i] for key in labels.keys()}

                # Get valid length
                valid_steps = int(sample_labels["month_length"].item() + 28) * 96

                # Flatten but keep full length
                ts_flat = profile.flatten()

                # Create validity mask
                validity_mask = torch.ones(len(ts_flat))
                validity_mask[valid_steps:] = 0.0

                if self.task_type == "imputation":
                    # Create training mask for imputation
                    if self.imputation_type == "mcar":
                        training_mask = generate_mcar_mask(
                            len(ts_flat),
                            missing_rate=self.missing_rate,
                            seed=42 + self._global_sample_idx,
                        )
                    elif self.imputation_type == "mnar_consecutive":
                        training_mask = generate_mnar_consecutive_mask(
                            len(ts_flat),
                            missing_rate=self.missing_rate,
                            seed=42 + self._global_sample_idx,
                        )
                    training_mask = training_mask * validity_mask
                    batch_masks.append(training_mask)
                    batch_target_data.append(ts_flat)

                elif self.task_type == "super_resolution":
                    # For SR, create downsampled version as input and original as target
                    # Downsample: average every scale_factor points
                    ts_downsampled = ts_flat.reshape(-1, self.scale_factor).mean(dim=1)

                    # Store both low-res and high-res versions
                    batch_downsampled.append(ts_downsampled)  # Low-res as input
                    batch_target_data.append(ts_flat)  # High-res as target

                if self.task_type != "super_resolution":
                    raise ValueError("invalid task type")

                batch_validity.append(validity_mask)
                self._global_sample_idx += 1

            if self.task_type == "imputation":
                yield (
                    torch.stack(batch_target_data),
                    torch.stack(batch_masks),
                    torch.stack(batch_validity),
                )
            else:  # super_resolution
                yield (
                    torch.stack(batch_downsampled),
                    torch.stack(batch_target_data),
                    torch.stack(batch_validity),
                )  # low_res, high_res

    def __len__(self):
        return len(self.original_dataloader)


ModelFormat = Literal["2d", "3d", "valid_only", "3d_paired"]


class BaselineProcessedDataLoader:
    """DataLoader that yields processed data for specific baseline models."""

    def __init__(
        self, base_dataloader, model_format: ModelFormat = "2d", ignore_validity=False
    ):
        """
        Args:
            base_dataloader: BaselineDataLoader instance
            model_format: "2d" (KNN), "3d" (BRITS/MaskedAE), or "valid_only" (GP/CNN - extract valid positions)
            ignore_validity: If True, treat all positions as valid (for fixed-length models like KNN)
        """
        self.base_dataloader = base_dataloader
        self.model_format = model_format
        self.ignore_validity = ignore_validity

    def __iter__(self):
        for batch in self.base_dataloader:
            if self.base_dataloader.task_type == "imputation":
                ts_batch, mask_batch, validity_batch = batch

                if self.model_format == "valid_only" and not self.ignore_validity:
                    # For GP/CNN - extract only valid positions
                    processed_ts = []
                    processed_masks = []

                    for i in range(ts_batch.shape[0]):
                        ts = ts_batch[i]
                        mask = mask_batch[i]
                        validity = validity_batch[i]

                        valid_indices = validity.bool()
                        ts_valid = ts[valid_indices].unsqueeze(0)
                        mask_valid = mask[valid_indices].unsqueeze(0)

                        processed_ts.append(ts_valid)
                        processed_masks.append(mask_valid)

                    yield processed_ts, processed_masks

                elif self.model_format == "3d":
                    # For BRITS/MaskedAE - convert to 3D
                    ts_3d = ts_batch.unsqueeze(-1)  # [B, T, 1]
                    mask_3d = mask_batch.unsqueeze(-1)  # [B, T, 1]
                    yield ts_3d, mask_3d

                else:  # 2d - for KNN with ignore_validity=True
                    if self.ignore_validity:
                        # KNN: use full tensors, ignore validity (90%+ are valid anyway)
                        yield ts_batch, mask_batch
                    else:
                        yield ts_batch, mask_batch, validity_batch

            elif self.base_dataloader.task_type == "super_resolution":
                low_res_batch, high_res_batch, validity_batch = batch

                if self.model_format == "valid_only":
                    # For GP - return lists of variable-length tensors (already processed)
                    low_res_list = []
                    high_res_list = []

                    processed_ts = []
                    processed_masks = []

                    for i in range(low_res_batch.shape[0]):
                        ts_low = low_res_batch[i]
                        ts_high = high_res_batch[i]
                        validity = validity_batch[i]

                        valid_indices = validity.bool()
                        ts_low_valid = ts_low[valid_indices].unsqueeze(0)
                        ts_high_valid = ts_high[valid_indices].unsqueeze(0)

                        low_res_list.append(ts_low_valid)
                        high_res_list.append(ts_high_valid)

                    yield low_res_list, high_res_list

                elif self.model_format == "3d":
                    # For CNN - convert to 3D [B, T, 1] format
                    low_res_3d = low_res_batch.unsqueeze(-1)
                    high_res_3d = high_res_batch.unsqueeze(-1)
                    yield low_res_3d, high_res_3d

                else:  # 2d format
                    yield low_res_batch, high_res_batch
            else:
                raise ValueError(
                    f"Unsupported task type: {self.base_dataloader.task_type}"
                )

    def __len__(self):
        return len(self.base_dataloader)
