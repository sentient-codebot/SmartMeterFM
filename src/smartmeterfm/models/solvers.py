"""Custom flow-matching ODE solvers for SmartMeterFM.

This module provides ``FMSolver``, a drop-in replacement for
``flow_matching.solver.ODESolver`` that additionally supports:

- non-uniform time grids (``"uniform"`` or ``"geometric"``);
- RePaint-style inner resampling for posterior sampling
  (``resample_steps`` inner refinements at ``t < resample_t_threshold``);
- valid-length zero-padding between inner refinements when the model uses
  per-sample masking.

When ``resample_steps == 0`` and ``time_grid_mode == "uniform"`` the solver
degrades to plain Euler integration and is byte-equivalent (up to FP
associativity) to ``ODESolver(method="euler", step_size=1/num_step)``.

State updates (``x = x + dt * v`` and the RePaint re-noising
``x = (1-t)*ε + t*x̂₁``) run in fp32 — the velocity ``v`` returned from the
wrapper chain is expected to be fp32 already (see
``smartmeterfm.models.flow.AutocastForwardWrapper`` for how bf16 is routed
surgically onto the inner NN forward only).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from flow_matching.utils import ModelWrapper
from torch import Tensor


def build_time_grid(
    n: int, mode: str, gamma: float, device: torch.device | str
) -> Tensor:
    """Return a strictly-increasing time grid of length ``n+1`` from 0 to 1.

    ``mode="uniform"`` → linspace (reproduces ``step_size=1/n``).
    ``mode="geometric"`` with ``gamma>1`` concentrates steps near ``t=0`` via
    ``t = u**gamma`` where ``u`` is uniform on ``[0,1]`` — useful for
    posterior sampling where x̂₁ is most uncertain at small ``t``. Early gaps
    are small (many fine steps where it matters), late gaps are larger.
    """
    u = torch.linspace(0.0, 1.0, n + 1, device=device)
    if mode == "uniform":
        return u
    if mode == "geometric":
        return u.pow(gamma)
    raise ValueError(f"Unknown time_grid_mode: {mode!r}")


def _zero_pad_invalid(x: Tensor, valid_length: Tensor) -> Tensor:
    """Zero out padded positions in ``x`` beyond each sample's valid length.

    Mirrors ``FlowModelPL._zero_padding`` — ``valid_length`` is given in
    FULL (unfolded) timestep units; the patched tensor's seq axis is
    masked at ``valid_length // num_ch``. Re-implemented here to keep
    ``solvers.py`` light (no import of the lightning module).

    Args:
        x: ``[B, L, D]`` patched tensor (D = ``num_in_channel`` = patch size).
        valid_length: ``[B]`` valid length in full unfolded timestep units.
    """
    folded_len = x.shape[1]
    num_ch = x.shape[2]
    valid_folded = valid_length.to(x.device) // num_ch
    mask = torch.arange(folded_len, device=x.device).unsqueeze(
        0
    ) < valid_folded.unsqueeze(1)
    return x * mask.unsqueeze(-1)


class FMSolver:
    """Flow-matching ODE solver with optional posterior-style inner
    refinement.

    Drop-in replacement for ``flow_matching.solver.ODESolver`` for the
    Euler method with the additional features described in the module
    docstring.

    Parameters
    ----------
    velocity_model:
        A ``ModelWrapper`` or callable that maps ``(x, t)`` to a velocity
        tensor of the same shape as ``x``. The caller is responsible for
        any conditioning, classifier-free guidance, posterior projection,
        or autocast — those live in the wrapper chain (see
        ``smartmeterfm.models.flow``).
    """

    def __init__(self, velocity_model: ModelWrapper | Callable):
        self.velocity_model = velocity_model

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        *,
        num_step: int,
        method: str = "euler",
        time_grid_mode: str = "uniform",
        time_grid_gamma: float = 2.0,
        resample_steps: int = 0,
        resample_t_threshold: float = 0.4,
        valid_length: Tensor | None = None,
    ) -> Tensor:
        """Integrate ``dx/dt = v(x, t)`` from ``t=0`` (``x_init``) to ``t=1``.

        Parameters
        ----------
        x_init:
            Initial state ``[B, ...]`` at ``t=0`` (typically Gaussian noise
            in flow-matching).
        num_step:
            Number of outer integration steps (NFE for ``method="euler"``
            with ``resample_steps=0``).
        method:
            Integration method. Currently only ``"euler"`` is supported.
        time_grid_mode:
            ``"uniform"`` (default) or ``"geometric"``. See
            ``build_time_grid``.
        time_grid_gamma:
            Exponent for ``"geometric"`` mode. Ignored otherwise.
        resample_steps:
            ``K``: number of RePaint-style inner refinement iterations to
            perform whenever ``t < resample_t_threshold``. Set to ``0`` for
            plain Euler.
        resample_t_threshold:
            ``τ``: only do inner refinement on outer steps with ``t < τ``.
            Ignored if ``resample_steps == 0``.
        valid_length:
            Optional ``[B]`` tensor of valid lengths (in flattened
            sequence units) for per-sample zero-padding between inner
            refinements. Ignored if ``resample_steps == 0``.

        Returns
        -------
        Tensor
            Final state ``[B, ...]`` at ``t=1``.
        """
        if method != "euler":
            raise NotImplementedError(
                f"FMSolver currently supports only method='euler'; got {method!r}."
            )

        grid = build_time_grid(
            num_step, time_grid_mode, time_grid_gamma, device=x_init.device
        )
        K = resample_steps
        tau = resample_t_threshold
        x = x_init
        for i in range(num_step):
            t_i = grid[i]
            dt = (grid[i + 1] - t_i).item()
            t_i_scalar = t_i.item()
            one_minus_t = 1.0 - t_i_scalar

            # RePaint-style inner refinement at the uncertain early regime.
            if K > 0 and t_i_scalar < tau:
                for _ in range(K):
                    v = self.velocity_model(x, t_i)
                    v = v.float() if v.dtype != x.dtype else v
                    x1_hat = x + one_minus_t * v
                    eps = torch.randn_like(x)
                    x = one_minus_t * eps + t_i_scalar * x1_hat
                    if valid_length is not None:
                        x = _zero_pad_invalid(x, valid_length)

            # Outer Euler step.
            v = self.velocity_model(x, t_i)
            v = v.float() if v.dtype != x.dtype else v
            x = x + dt * v
        return x
