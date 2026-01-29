"""Scaling-based estimation"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import brentq
from tqdm import tqdm


@dataclass
class AverageProfileState:
    n: int = 0
    profile: torch.Tensor = None  # [L, D]

    def update(self, new_profile: torch.Tensor):  # [L, D]
        if self.profile is None:
            self.profile = new_profile
            self.n = 1
        else:
            self.profile = (self.n * self.profile + new_profile) / (self.n + 1)
            self.n += 1


def find_param_a(normalized_profile: torch.Tensor, max: float, total: float, bounds: tuple|list|None = None):
    """
    Find the parameter a such that the normalized profile matches the given max and total.
    """
    normalized_profile = normalized_profile.cpu().numpy().flatten()  # [L*D,]
    bounds = bounds or (0, 100)
    def equation(a):
        return np.sum(np.power(normalized_profile, a)) - total / max

    # Use brentq to find the root of the equation
    try:
        a = brentq(equation, *bounds)  # Adjust bounds as necessary
    except ValueError:  # if not found opposite sign
        try:
            _bounds = (bounds[0]*0.1, bounds[1]*10)
            a = brentq(equation, *_bounds)
        except Exception as e:
            raise ValueError(f"max {max}, total {total}, bounds {bounds} -> {_bounds}; brentq raised Exception") from e
    return a


def _set_month_to_profile_dict():
    return defaultdict(AverageProfileState)


def _set_year_month_to_profile_dict():
    return defaultdict(_set_month_to_profile_dict)


def _set_category_year_month_to_profile_dict():
    return defaultdict(_set_year_month_to_profile_dict)


class Scaling:
    """ """

    def __init__(self):
        self.average_profile_dictionary = _set_category_year_month_to_profile_dict()

    def train(self, dataloader):
        for profile, labels in tqdm(dataloader):
            batch_size = profile.shape[0]
            for idx in range(batch_size):
                single_profile = profile[idx]
                baseload_profile = int(labels["BASELOAD_PROFILE"][idx].item())
                year = int(labels["year"][idx].item())
                month = int(labels["month"][idx].item())
                self.average_profile_dictionary[baseload_profile][year][month].update(
                    single_profile
                )

    def _get_avg_profile(self, baseload_profile: int, year: int, month: int):
        category = baseload_profile
        avg_profile = self.average_profile_dictionary[category][year][month].profile

        return avg_profile

    def _scaling_factor_check(self, max: float | None, total: float | None):
        if max is not None and total is not None:
            if max > total:
                raise ValueError(f"Max value {max:.3f} must be no greater than total value {total:.3f}.")

    def estimate(
        self,
        baseload_profile: int,
        year: int,
        month: int,
        max: float | None = None,
        total: float | None = None,
        bounds: tuple | list | None = None,
    ):
        # self._scaling_factor_check(max, total)
        avg_profile = self._get_avg_profile(baseload_profile, year, month)  # [L, D]
        if max is not None and total is None:
            # normalize by max
            normalized_profile = avg_profile / avg_profile.max().clamp(
                min=1e-6
            )  # [L, D]
            return normalized_profile * max  # [L, D]
        elif max is None and total is not None:
            # normalize by total
            normalizer = avg_profile.sum()
            normalizer = torch.where(
                normalizer > 0,
                normalizer.clamp(min=1e-6),
                normalizer.clamp(max=1e-6),
            )
            normalized_profile = avg_profile / normalizer  # [L, D]
            return normalized_profile * total  # [L, D]
        elif max is not None and total is not None:
            normalized_profile = avg_profile - avg_profile.min()  #  >= 0
            normalized_profile = normalized_profile / normalized_profile.max().clamp(
                min=1e-6
            )  # [L, D]
            if abs(max) <= 1e-5 and abs(total) <= 1e-5:
                return torch.zeros_like(normalized_profile)

            param_a = find_param_a(normalized_profile, max, total, bounds=bounds)
            return normalized_profile**param_a * max  # [L, D]
        else:
            raise ValueError("Either max or total must be provided.")
