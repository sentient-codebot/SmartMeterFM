"""PyTorch Dataset wrapper for time series profiles."""

from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


class Dataset1D(Dataset):
    def __init__(
        self,
        profile: Float[Tensor, "batch sequence channel"],
        label: Float[Tensor, "batch *"] | None = None,
    ):
        super().__init__()
        self.profile = profile.clone()
        self.label = label.clone() if label is not None else None

    def __len__(self):
        return len(self.profile)

    def __getitem__(self, idx):
        profile = self.profile[idx].clone()
        if self.label is not None:
            label = self.label[idx].clone()
            return profile, label
        return profile, None

    @property
    def num_channel(self):
        return self.profile.shape[2]

    @property
    def sequence_length(self):
        return self.profile.shape[1]

    @property
    def sample_shape(self):
        return (self.sequence_length, self.num_channel)

    def __repr__(self):
        return f"Dataset1D(profile={self.profile.shape})"
