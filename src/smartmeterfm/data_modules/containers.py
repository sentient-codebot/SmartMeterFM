"""Data container classes for profiles and labels."""

import enum
from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor


class StaticLabelContainer:
    def __init__(self, dict_labels: dict[str, Tensor], output_type: str = "tensor"):
        """statis labels. container for label names and tensor values.
        - also defines their operation (addition, selection)
        - each label is of shape [num_sample, feature_dim]

        Args:
            - dict_labels: dict of label names and tensor values
                tensor shape: [num_sample, feature_dim]
            - output_type: str, "tensor" or "dict". decided the type of __getitem__.

        Access the label tensor by:
            - label_container["label_name"] or\
                label_container["label_name1", "label_name2", ...]
            -> returns concatenated tensor of the labels\
                [num_sample, feature_dim1 + feature_dim2 + ...]
        """
        super().__init__()
        self.dict_labels = dict_labels
        self.output_type = output_type

        # checks
        if output_type not in ["tensor", "dict"]:
            raise ValueError("output_type must be 'tensor' or 'dict'")
        if not len(self.label_values) == 0:  # if non empty:
            if not (
                all(value.ndim == 2 for value in self.label_values)
                or (all(value.ndim == 1 for value in self.label_values))
            ):
                raise ValueError("inconsistent dimensions among labels.")
            if self.label_values[0].ndim == 2 and not all(
                value.shape[0] == self.label_values[0].shape[0]
                for value in self.label_values
            ):
                raise ValueError("inconsistent batch size among labels.")

    def __repr__(self) -> str:
        return f"StaticLabelContainer({self.label_names})"

    @property
    def label_names(self) -> list[str]:
        return list(self.dict_labels.keys())

    @property
    def label_values(self) -> list[Tensor]:
        return list(self.dict_labels.values())

    def __add__(self, other: "StaticLabelContainer") -> "StaticLabelContainer":
        assert all(
            _other_name not in self.label_names for _other_name in other.label_names
        )
        return StaticLabelContainer({**self.dict_labels, **other.dict_labels})

    def __contains__(self, key: str) -> bool:
        return key in self.dict_labels

    def __getitem__(self, key: str | Sequence[str]) -> Tensor | dict[str, Tensor]:
        if self.output_type == "tensor":
            if isinstance(key, str):
                return self.dict_labels[key].clone()
            elif isinstance(key, Sequence):
                return torch.cat(
                    [self.dict_labels[_key] for _key in key], dim=-1
                ).clone()
        else:
            if isinstance(key, str):
                return {key: self.dict_labels[key].clone()}
            elif isinstance(key, Sequence):
                return {_key: self.dict_labels[_key].clone() for _key in key}


class DatasetWithMetadata(NamedTuple):
    """Storage for profile and labels
    Args:
        - profile: dict of [task_name: profile_tensor]
        - label: dict of [task_name: StaticLabelContainer]
        - scaling_factor: optional, tuple of (mean, std) or (max, min) for\
            normalization
        - pit: optional, callable function for PIT

    Example:
        >>> profile = {
            "train": torch.randn(32, 180, 8),
        }
        >>> label = {
            "train": StaticLabelContainer({
                "label1": torch.randn(32, 1),
                "label2": torch.randn(32, 2),
            }),
        }
    """

    profile: dict[str, Float[Tensor, "batch sequence channel"]]
    label: dict[str, StaticLabelContainer]
    scaling_factor: tuple[Tensor, Tensor] | None = None
    pit: Callable[[Tensor], Tensor] | None = None


class DataShapeType(enum.Enum):
    """
    Enum class for data shape
    """

    BATCH_CHANNEL_SEQUENCE = enum.auto()
    BATCH_SEQUENCE_CHANNEL = enum.auto()
    BATCH_SEQUENCE = enum.auto()
    CHANNEL_SEQUENCE = enum.auto()
    SEQUENCE = enum.auto()
