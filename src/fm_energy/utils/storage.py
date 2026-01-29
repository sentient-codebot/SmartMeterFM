"""tools for facilitating task-specific sample storage."""

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


class HDF5SampleStorage:
    """Hierarchical storage for generated samples with task/category/date/cluster structure.

    Manages storage layout: task -> category -> date -> cluster -> samples
    Each cluster can contain multiple sample batches that are efficiently stored.
    """

    _MERGED_PROCESS_ATTR = "merged_process_ids"

    def __init__(
        self,
        file_path: str | Path,
        mode: str = "a",
        process_id: int = None,
        swmr: bool = False,
    ):
        """Initialize HDF5 storage with optional process-aware file naming.

        Args:
            file_path: Path to HDF5 file
            mode: File access mode ('r', 'w', 'a'). Default 'a' for append/create.
            process_id: Process ID for parallel writing. If provided, creates process-specific files.
            swmr: If True, open the file in single-writer/multi-reader mode.
        """
        self.original_file_path = Path(file_path)
        self.process_id = process_id
        self.mode = mode
        self.swmr = swmr
        self._file = None
        self._is_writer = mode != "r"

        # Determine actual file path for this instance
        if process_id is not None:
            # Process-specific file naming
            stem = self.original_file_path.stem
            suffix = self.original_file_path.suffix
            self.file_path = (
                self.original_file_path.parent / f"{stem}_proc{process_id}{suffix}"
            )
            print(f"🔧 HDF5Storage: Using process-specific file: {self.file_path}")
        else:
            self.file_path = self.original_file_path

    def __enter__(self):
        open_kwargs: dict[str, Any] = {}
        if self.swmr:
            open_kwargs["libver"] = "latest"
            if not self._is_writer:
                open_kwargs["swmr"] = True

        try:
            self._file = h5py.File(self.file_path, self.mode, **open_kwargs)
        except (OSError, ValueError) as exc:
            if self.swmr:
                raise RuntimeError(
                    "Failed to open HDF5 file in SWMR mode. Ensure the file was created "
                    "with libver='latest' or disable swmr=True."
                ) from exc
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            self._file = None

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file:
            self._file.close()
            self._file = None

    def _flush_after_write(self) -> None:
        """Flush pending writes and activate SWMR write mode when enabled."""
        if not self.swmr or not self._is_writer:
            return
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")
        self._file.flush()
        if not self._file.swmr_mode:
            try:
                self._file.swmr_mode = True
            except (OSError, AttributeError) as exc:
                raise RuntimeError(
                    "Failed to enable SWMR write mode after flushing changes."
                ) from exc

    def _ensure_group_path(
        self, task: str, category: str, date: str, cluster: int
    ) -> h5py.Group:
        """Ensure hierarchical group structure exists and return cluster group."""
        task_group = self._file.require_group(task)
        cat_group = task_group.require_group(category)
        date_group = cat_group.require_group(date)
        cluster_group = date_group.require_group(f"cluster_{cluster}")
        return cluster_group

    def add_samples(
        self,
        task: str,
        category: str,
        date: str,
        cluster: int,
        samples: torch.Tensor | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ):
        """Add generated samples to storage.

        Continuously store batch of samples under task/category/date/cluster structure. New samples are added as new datasets with name "samples_{next_batch_idx}".

        Args:
            task: Task name (e.g., "SR_2x", "SR_4x")
            category: Category name (e.g., "E3A_NO_GEN")
            date: Date identifier (e.g., "2022_0")
            cluster: Cluster ID (int)
            samples: Generated samples tensor [N, ...]
            metadata: Optional metadata dict.
        """
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")

        cluster_group = self._ensure_group_path(task, category, date, cluster)

        # Convert to numpy if tensor
        if isinstance(samples, torch.Tensor):
            samples_np = samples.cpu().numpy()
        elif isinstance(samples, np.ndarray):
            samples_np = samples
        else:
            samples_np = np.array(samples)

        # Find next available sample batch index
        existing_keys = [k for k in cluster_group.keys() if k.startswith("samples_")]
        if existing_keys:
            max_idx = max(int(k.split("_")[1]) for k in existing_keys)
            next_idx = max_idx + 1
        else:
            next_idx = 0

        # Store samples
        dataset_name = f"samples_{next_idx}"
        cluster_group.create_dataset(dataset_name, data=samples_np, compression="gzip")

        # Default metadata: creation time
        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)

        if "metadata" not in cluster_group:
            metadata.setdefault("created_at", datetime.now().isoformat())
        # Store metadata if provided
        if metadata:
            meta_group = cluster_group.require_group("metadata")
            self._write_cluster_metadata(
                meta_group=meta_group,
                metadata=metadata,
                task=task,
                category=category,
                date=date,
                cluster=cluster,
            )

        self._flush_after_write()

    def add_mask(
        self,
        task: str,
        category: str,
        date: str,
        cluster: int,
        mask: torch.Tensor | np.ndarray,
    ):
        """Add mask data for a cluster.

        Args:
            task: Task name
            category: Category name
            date: Date identifier
            cluster: Cluster ID
            mask: Mask tensor/array
        """
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")

        cluster_group = self._ensure_group_path(task, category, date, cluster)

        # Convert to numpy if tensor
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Remove existing mask if present
        if "mask" in cluster_group:
            del cluster_group["mask"]

        cluster_group.create_dataset("mask", data=mask_np, compression="gzip")
        # print(f"💾 Added mask for {task}/{category}/{date}/cluster_{cluster}")
        self._flush_after_write()

    def copy_cluster_from(
        self,
        source_group: h5py.Group,
        task: str,
        category: str,
        date: str,
        cluster: int,
        *,
        allow_overwrite: bool = False,
        merge_existing: bool = False,
    ) -> None:
        """Copy an entire cluster group from another HDF5 file into this storage.

        Args:
            source_group: h5py group pointing to ``cluster_<idx>`` in the source file.
            task: Target task name.
            category: Target category name.
            date: Target date identifier.
            cluster: Target cluster identifier.
            allow_overwrite: If False and the target cluster already exists, raise an error.
            merge_existing: When True, merge new datasets into the existing cluster instead
                of replacing it entirely. Requires ``allow_overwrite=True``.
        """
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")
        if not isinstance(source_group, h5py.Group):
            raise TypeError("source_group must be an h5py.Group instance.")
        if merge_existing and not allow_overwrite:
            raise ValueError("merge_existing=True requires allow_overwrite=True.")

        task_group = self._file.require_group(task)
        category_group = task_group.require_group(category)
        date_group = category_group.require_group(date)
        cluster_name = f"cluster_{cluster}"

        if cluster_name in date_group:
            if not allow_overwrite:
                raise FileExistsError(
                    f"Cluster {task}/{category}/{date}/{cluster_name} already exists."
                )
            target_group = date_group[cluster_name]
            if merge_existing:
                self._merge_group_contents(target_group, source_group)
            else:
                del date_group[cluster_name]
                self._file.copy(source_group, date_group, cluster_name)
        else:
            self._file.copy(source_group, date_group, cluster_name)

        self._flush_after_write()

    def _merge_group_contents(
        self, target_group: h5py.Group, source_group: h5py.Group
    ) -> None:
        """Merge datasets and attributes from source_group into target_group."""
        for attr_name, attr_value in source_group.attrs.items():
            target_group.attrs[attr_name] = attr_value

        for key, item in source_group.items():
            if key in target_group:
                raise ValueError(
                    f"Dataset or subgroup '{key}' already exists in target group."
                )
            self._file.copy(item, target_group, name=key)

    @staticmethod
    def merge_process_files(
        original_file_path: str | Path,
        num_processes: int,
        cleanup: bool = True,
        append_mode: bool = None,
    ) -> bool:
        """Merge process-specific HDF5 files into a single file.

        Args:
            original_file_path: Original file path (without process suffix)
            num_processes: Number of process files to merge
            cleanup: If True, delete individual process files after merging
            append_mode: If True, append to existing file. If None, auto-detect based on file existence.

        Returns:
            True if merge successful, False otherwise
        """
        original_path = Path(original_file_path)
        print(f"🔗 Merging {num_processes} process files into {original_path}")

        # Auto-detect append mode if not specified
        if append_mode is None:
            append_mode = original_path.exists()
            if append_mode:
                print(
                    "📋 Target file exists - using append mode to preserve existing data"
                )
            else:
                print("📁 Target file doesn't exist - creating new file")
        elif append_mode and not original_path.exists():
            print(
                "📁 Append mode requested but target doesn't exist - creating new file"
            )
            append_mode = False

        file_mode = "a" if append_mode else "w"

        try:
            with h5py.File(original_path, file_mode) as final_h5:
                files_merged = 0

                merged_ids = HDF5SampleStorage._load_merged_process_ids(final_h5)

                for proc_id in range(num_processes):
                    stem = original_path.stem
                    suffix = original_path.suffix
                    proc_file = original_path.parent / f"{stem}_proc{proc_id}{suffix}"

                    if not proc_file.exists():
                        print(f"⚠️ Process file {proc_file} not found - skipping")
                        continue

                    if proc_id in merged_ids:
                        print(
                            f"ℹ️ Process {proc_id} already merged previously - skipping {proc_file}"
                        )
                        if cleanup:
                            try:
                                proc_file.unlink()
                                print(f"🗑️ Cleaned up {proc_file}")
                            except Exception as cleanup_exc:
                                print(f"⚠️ Could not remove {proc_file}: {cleanup_exc}")
                        continue

                    print(f"🔗 Merging process {proc_id} file...")

                    with h5py.File(proc_file, "r") as proc_h5:
                        conflicts = HDF5SampleStorage._detect_dataset_conflicts(
                            final_h5, proc_h5
                        )
                        if conflicts:
                            conflict_preview = ", ".join(conflicts[:5])
                            if len(conflicts) > 5:
                                conflict_preview += ", ..."
                            raise RuntimeError(
                                "Merge aborted: destination already contains dataset(s) "
                                f"from process {proc_id}: {conflict_preview}"
                            )

                        # Copy all groups and datasets recursively
                        def copy_recursively(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                # Copy dataset with all attributes
                                final_h5.copy(obj, name)
                            elif isinstance(obj, h5py.Group):
                                # Create group if it doesn't exist
                                if name not in final_h5:
                                    grp = final_h5.create_group(name)
                                    # Copy group attributes
                                    for attr_name, attr_val in obj.attrs.items():
                                        grp.attrs[attr_name] = attr_val

                        proc_h5.visititems(copy_recursively)

                    merged_ids.add(proc_id)
                    HDF5SampleStorage._store_merged_process_ids(final_h5, merged_ids)
                    final_h5.flush()
                    files_merged += 1

                    # Clean up individual file
                    if cleanup:
                        try:
                            proc_file.unlink()
                            print(f"🗑️ Cleaned up {proc_file}")
                        except Exception as e:
                            print(f"⚠️ Could not remove {proc_file}: {e}")

            print(f"✅ Successfully merged {files_merged} files into {original_path}")
            return True

        except Exception as e:
            print(f"❌ Error merging files: {e}")
            return False

    def get_samples(
        self,
        task: str,
        category: str,
        date: str,
        cluster: int,
        batch_idx: int | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """Retrieve samples from storage.

        Args:
            task: Task name
            category: Category name
            date: Date identifier
            cluster: Cluster ID
            batch_idx: Specific batch index. If None, returns all batches.

        Returns:
            Single numpy array if batch_idx specified, list of arrays otherwise.
        """
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")

        try:
            cluster_group = self._file[task][category][date][f"cluster_{cluster}"]
        except KeyError as e:
            raise KeyError(
                f"Path not found: {task}/{category}/{date}/cluster_{cluster}"
            ) from e

        sample_keys = [k for k in cluster_group.keys() if k.startswith("samples_")]
        sample_keys.sort(key=lambda x: int(x.split("_")[1]))

        if not sample_keys:
            raise ValueError(
                f"No samples found for {task}/{category}/{date}/cluster_{cluster}"
            )

        if batch_idx is not None:
            dataset_name = f"samples_{batch_idx}"
            if dataset_name not in cluster_group:
                raise KeyError(f"Batch {batch_idx} not found")
            return cluster_group[dataset_name][:]
        else:
            return [cluster_group[key][:] for key in sample_keys]

    def get_all_samples(
        self, task: str, category: str, date: str, cluster: int
    ) -> np.ndarray:
        """Get all samples for a cluster concatenated along batch dimension.

        Returns:
            Concatenated numpy array of all sample batches.
        """
        sample_batches = self.get_samples(task, category, date, cluster)
        return np.concatenate(sample_batches, axis=0)

    def get_metadata(
        self, task: str, category: str, date: str, cluster: int
    ) -> dict[str, Any]:
        """Retrieve metadata for a cluster.

        Returns:
            Dictionary of metadata keys and values.
        """
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")

        try:
            cluster_group = self._file[task][category][date][f"cluster_{cluster}"]
            if "metadata" not in cluster_group:
                return {}

            meta_group = cluster_group["metadata"]
            metadata = dict(meta_group.attrs)

            # Add dataset metadata
            for key in meta_group.keys():
                metadata[key] = meta_group[key][:]

            return metadata
        except KeyError:
            return {}

    def list_contents(self) -> dict[str, Any]:
        """List all available tasks, categories, dates, and clusters.

        Returns:
            Nested dictionary showing storage structure.
        """
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")

        contents = {}
        for task_name in self._file.keys():
            contents[task_name] = {}
            task_group = self._file[task_name]

            for cat_name in task_group.keys():
                contents[task_name][cat_name] = {}
                cat_group = task_group[cat_name]

                for date_name in cat_group.keys():
                    contents[task_name][cat_name][date_name] = []
                    date_group = cat_group[date_name]

                    for cluster_name in date_group.keys():
                        if cluster_name.startswith("cluster_"):
                            cluster_id = int(cluster_name.split("_")[1])
                            contents[task_name][cat_name][date_name].append(cluster_id)

                    contents[task_name][cat_name][date_name].sort()

        return contents

    def get_mask(
        self,
        task: str,
        category: str,
        date: str,
        cluster: int,
    ) -> np.ndarray | None:
        """Retrieve mask for a cluster.

        Args:
            task: Task name
            category: Category name
            date: Date identifier
            cluster: Cluster ID

        Returns:
            Mask array if exists, None otherwise.
        """
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")

        try:
            cluster_group = self._file[task][category][date][f"cluster_{cluster}"]
            if "mask" not in cluster_group:
                return None
            return cluster_group["mask"][:]
        except KeyError:
            return None

    def get_sample_count(
        self, task: str, category: str, date: str, cluster: int
    ) -> int:
        """Get total number of samples stored for a cluster."""
        try:
            all_samples = self.get_all_samples(task, category, date, cluster)
            return len(all_samples)
        except (KeyError, ValueError):
            return 0

    def task_exists(self, task: str) -> bool:
        """Check if a task group exists in the file.

        Args:
            task: Task name to check (e.g., "SR_2x", "SR_4x")

        Returns:
            True if task group exists, False otherwise
        """
        if self._file is None:
            # Check if file exists at all
            return self.file_path.exists() and self._check_task_in_file(task)
        return task in self._file

    def _check_task_in_file(self, task: str) -> bool:
        """Helper to check task existence when file is not open."""
        try:
            with h5py.File(self.file_path, "r") as f:
                return task in f
        except OSError:
            return False

    @staticmethod
    def validate_generation_target(
        file_path: str | Path, task: str, append_mode: bool
    ) -> None:
        """Validate generation target before starting generation.

        Args:
            file_path: Path to target HDF5 file
            task: Task name (e.g., "SR_2x", "SR_4x")
            append_mode: Whether append mode is enabled

        Raises:
            RuntimeError: If task exists and append_mode is False
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"📁 Target file doesn't exist - will create new file: {file_path}")
            return

        # Check if task already exists
        try:
            with h5py.File(file_path, "r") as f:
                if task in f:
                    if not append_mode:
                        raise RuntimeError(
                            f"❌ Task '{task}' already exists in {file_path}!\n"
                            f"This would overwrite existing data. Options:\n"
                            f"  1. Use --append flag to add more samples to existing task\n"
                            f"  2. Use a different output file\n"
                            f"  3. Remove existing task data manually"
                        )
                    else:
                        print(
                            f"📋 Task '{task}' exists - will append new samples (append mode)"
                        )
                else:
                    print(f"📋 Task '{task}' doesn't exist - will add new task group")
        except OSError as e:
            print(f"⚠️ Could not read existing file {file_path}: {e}")
            print(
                "🔄 Proceeding with generation (assuming file is corrupted or inaccessible)"
            )

    def _write_cluster_metadata(
        self,
        meta_group: h5py.Group,
        metadata: dict[str, Any],
        *,
        task: str,
        category: str,
        date: str,
        cluster: int,
    ) -> None:
        """Persist metadata for a cluster while enforcing immutability across batches."""
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                new_value = self._normalize_scalar_metadata(value)
                if key in meta_group.attrs:
                    stored_value = self._normalize_scalar_metadata(
                        meta_group.attrs[key]
                    )
                    if stored_value != new_value:
                        raise ValueError(
                            "Metadata mismatch for "
                            f"{task}/{category}/{date}/cluster_{cluster}: key '{key}' "
                            f"already set to {stored_value!r}, new value {new_value!r} differs."
                        )
                else:
                    meta_group.attrs[key] = value
                continue

            value_np = self._to_numpy_metadata(value, key)

            if key in meta_group:
                existing_np = meta_group[key][:]
                if not self._metadata_arrays_equal(existing_np, value_np):
                    raise ValueError(
                        "Metadata mismatch for "
                        f"{task}/{category}/{date}/cluster_{cluster}: key '{key}' "
                        "already set to a different array value."
                    )
            else:
                meta_group.create_dataset(key, data=value_np, compression="gzip")

    @staticmethod
    def _normalize_scalar_metadata(value: Any) -> Any:
        """Normalize scalar metadata for reliable comparisons."""
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return value

    @staticmethod
    def _to_numpy_metadata(value: Any, key: str) -> np.ndarray:
        """Convert metadata value into numpy array form."""
        if isinstance(value, torch.Tensor):
            value_np = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            value_np = value
        else:
            try:
                value_np = np.asarray(value)
            except Exception as exc:
                raise TypeError(
                    f"invalid metadata type {type(value)} for field {key}"
                ) from exc

        if value_np.dtype == object:
            raise TypeError(
                f"metadata field {key} cannot be converted to a numeric/string array"
            )

        return value_np

    @staticmethod
    def _metadata_arrays_equal(existing: np.ndarray, new_value: np.ndarray) -> bool:
        """Check whether two metadata arrays should be considered identical."""
        if existing.shape != new_value.shape:
            return False

        if existing.dtype.kind in {"S", "U"} or new_value.dtype.kind in {"S", "U"}:
            existing_normalized = existing.astype("U")
            new_normalized = new_value.astype("U")
            return np.array_equal(existing_normalized, new_normalized)

        if np.issubdtype(existing.dtype, np.floating) or np.issubdtype(
            new_value.dtype, np.floating
        ):
            return np.allclose(existing, new_value, equal_nan=True)

        return np.array_equal(existing, new_value)

    @staticmethod
    def _load_merged_process_ids(file_handle: h5py.File) -> set[int]:
        """Load the set of process IDs that have already been merged."""
        attr = file_handle.attrs.get("merged_process_ids")
        if attr is None:
            return set()

        values = np.atleast_1d(attr)
        merged_ids: set[int] = set()
        for value in values:
            if isinstance(value, (bytes, bytearray)):
                merged_ids.add(int(value.decode("utf-8")))
            else:
                merged_ids.add(int(value))
        return merged_ids

    @staticmethod
    def _store_merged_process_ids(
        file_handle: h5py.File, process_ids: set[int]
    ) -> None:
        """Persist the merged process ID set onto the destination file."""
        if not process_ids:
            if "merged_process_ids" in file_handle.attrs:
                del file_handle.attrs["merged_process_ids"]
            return

        ids_array = np.array(sorted(process_ids), dtype=np.int64)
        file_handle.attrs["merged_process_ids"] = ids_array

    @staticmethod
    def _detect_dataset_conflicts(
        dest_handle: h5py.File, source_handle: h5py.File
    ) -> list[str]:
        """Return dataset paths that already exist in the destination file."""
        conflicts: list[str] = []

        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                if name in dest_handle:
                    conflicts.append(name)

        source_handle.visititems(visitor)
        return conflicts


class HDF5MetricsStorage:
    """Storage for evaluation metrics with task/model/category/date/cluster structure.

    Manages storage layout: metrics -> task -> model_size -> category -> date -> cluster -> metrics
    Supports appending new metrics to existing evaluations.
    """

    def __init__(self, file_path: str | Path, mode: str = "a", swmr: bool = False):
        """Initialize HDF5 metrics storage.

        Args:
            file_path: Path to HDF5 file
            mode: File access mode ('r', 'w', 'a'). Default 'a' for append/create.
        """
        self.file_path = Path(file_path)
        self.mode = mode
        self.swmr = swmr
        self._file = None
        self._is_writer = mode != "r"

    def __enter__(self):
        open_kwargs: dict[str, Any] = {}
        if self.swmr:
            open_kwargs["libver"] = "latest"
            if not self._is_writer:
                open_kwargs["swmr"] = True

        try:
            self._file = h5py.File(self.file_path, self.mode, **open_kwargs)
        except (OSError, ValueError) as exc:
            if self.swmr:
                raise RuntimeError(
                    "Failed to open HDF5 file in SWMR mode. Ensure the file was created "
                    "with libver='latest' or disable swmr=True."
                ) from exc
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            self._file = None

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file:
            self._file.close()
            self._file = None

    def _flush_after_write(self) -> None:
        """Flush pending writes and activate SWMR write mode when enabled."""
        if not self.swmr or not self._is_writer:
            return
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")
        self._file.flush()
        if not self._file.swmr_mode:
            try:
                self._file.swmr_mode = True
            except (OSError, AttributeError) as exc:
                raise RuntimeError(
                    "Failed to enable SWMR write mode after flushing changes."
                ) from exc

    def _ensure_metrics_group_path(
        self, task: str, model_size: str, category: str, date: str, cluster: int
    ) -> h5py.Group:
        """Ensure hierarchical group structure exists for metrics."""
        metrics_group = self._file.require_group("metrics")
        task_group = metrics_group.require_group(task)
        model_group = task_group.require_group(model_size)
        cat_group = model_group.require_group(category)
        date_group = cat_group.require_group(date)
        cluster_group = date_group.require_group(f"cluster_{cluster}")
        return cluster_group

    def save_metrics(
        self,
        task: str,
        model_size: str,
        category: str,
        date: str,
        cluster: int,
        metrics: dict[str, Any],
        append: bool = True,
    ):
        """Save metrics for a specific task/model/category/date/cluster.

        Args:
            task: Task name (e.g., "SR_2x", "SR_4x")
            model_size: Model size (e.g., "xl", "large")
            category: Category name (e.g., "E3A_NO_GEN")
            date: Date identifier (e.g., "2022_0")
            cluster: Cluster ID (int)
            metrics: Dictionary of metric names to values
            append: If True, append to existing metrics. If False, overwrite.
        """
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")

        cluster_group = self._ensure_metrics_group_path(
            task, model_size, category, date, cluster
        )

        # Add timestamp if not present
        if "evaluated_at" not in metrics:
            metrics["evaluated_at"] = datetime.now().isoformat()

        # Save metrics as attributes (for simple scalars) or datasets (for arrays)
        for key, value in metrics.items():
            if isinstance(value, str, int, float, bool, np.number):
                if not append and key in cluster_group.attrs:
                    del cluster_group.attrs[key]
                cluster_group.attrs[key] = value
            else:
                # For arrays/complex data
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                elif not isinstance(value, np.ndarray):
                    value = np.array(value)

                if not append and key in cluster_group:
                    del cluster_group[key]
                cluster_group.create_dataset(key, data=value, compression="gzip")

        self._flush_after_write()

    def get_metrics(
        self,
        task: str,
        model_size: str,
        category: str,
        date: str,
        cluster: int,
    ) -> dict[str, Any]:
        """Retrieve metrics for a specific task/model/category/date/cluster.

        Returns:
            Dictionary of metric names to values.
        """
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")

        try:
            cluster_group = self._file["metrics"][task][model_size][category][date][
                f"cluster_{cluster}"
            ]

            metrics = dict(cluster_group.attrs)

            # Add dataset metrics
            for key in cluster_group.keys():
                metrics[key] = cluster_group[key][:]

            return metrics
        except KeyError:
            return {}

    def get_all_metrics_for_task(
        self, task: str, model_size: str
    ) -> dict[str, dict[str, Any]]:
        """Get all metrics for a specific task and model size.

        Returns:
            Dictionary with keys like "category/date/cluster_X" and values as metric dicts.
        """
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")

        try:
            task_group = self._file["metrics"][task][model_size]
            all_metrics = {}

            for category in task_group.keys():
                cat_group = task_group[category]
                for date in cat_group.keys():
                    date_group = cat_group[date]
                    for cluster_name in date_group.keys():
                        if cluster_name.startswith("cluster_"):
                            cluster_id = int(cluster_name.split("_")[1])
                            key = f"{category}/{date}/{cluster_name}"
                            all_metrics[key] = self.get_metrics(
                                task, model_size, category, date, cluster_id
                            )

            return all_metrics
        except KeyError:
            return {}

    def get_summary_stats(
        self, task: str, model_size: str, metric_name: str
    ) -> dict[str, float]:
        """Get summary statistics for a specific metric across all clusters.

        Args:
            task: Task name
            model_size: Model size
            metric_name: Name of the metric to summarize (e.g., "crps")

        Returns:
            Dictionary with mean, min, max, count statistics.
        """
        all_metrics = self.get_all_metrics_for_task(task, model_size)
        values = [
            metrics[metric_name]
            for metrics in all_metrics.values()
            if metric_name in metrics
        ]

        if not values:
            return {"mean": 0, "min": 0, "max": 0, "count": 0}

        return {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }

    def list_available_tasks(self) -> list[str]:
        """List all available tasks in the metrics storage."""
        if self._file is None:
            raise RuntimeError("Storage not open. Use context manager.")

        if "metrics" not in self._file:
            return []

        return list(self._file["metrics"].keys())
