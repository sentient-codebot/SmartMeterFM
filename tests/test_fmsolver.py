"""Tests for ``smartmeterfm.models.solvers.FMSolver``.

Coverage:

1. Bit-exact Euler equivalence with ``flow_matching.solver.ODESolver`` for
   the trivial case (``resample_steps=0``, ``time_grid_mode="uniform"``) —
   guarantees we can drop the third-party dependency without changing the
   default sampling trajectory.
2. ``build_time_grid`` shape / monotonicity / boundary behaviour.
3. RePaint-style inner refinement triggers exactly when ``t < tau`` and
   ``K > 0``, and zero-pads correctly when a ``valid_length`` is supplied.
4. ``AutocastForwardWrapper`` returns fp32 outputs even when the inner
   matmuls run in bf16.
"""

from __future__ import annotations

import torch
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from smartmeterfm.models.flow import AutocastForwardWrapper
from smartmeterfm.models.solvers import FMSolver, build_time_grid


class _AnalyticV(ModelWrapper):
    """Velocity field with a nontrivial closed form for equivalence tests."""

    def __init__(self):
        super().__init__(model=torch.nn.Identity())

    def forward(self, x, t, **_extras):
        # u(x, t) = -x + 0.3 * sin(2π t) — easy to integrate with Euler.
        return -x + 0.3 * torch.sin(
            2 * torch.pi * t.expand(x.shape[0]).view(-1, 1, 1)
        )


# --------------------------------------------------------------------------- #
# 1. Equivalence with flow_matching.solver.ODESolver                          #
# --------------------------------------------------------------------------- #


def test_fmsolver_euler_matches_odesolver_bitexact():
    """``FMSolver(num_step=N)`` ≡ ``ODESolver(step_size=1/N)`` to the bit."""
    torch.manual_seed(0)
    x0 = torch.randn(3, 5, 8)

    ode = ODESolver(velocity_model=_AnalyticV())
    x1_ref = ode.sample(x_init=x0, method="euler", step_size=1 / 64)

    fm = FMSolver(velocity_model=_AnalyticV())
    x1_new = fm.sample(x_init=x0, num_step=64, method="euler")

    assert x1_new.shape == x1_ref.shape
    assert torch.equal(x1_new, x1_ref), (
        f"FMSolver Euler diverges from ODESolver: "
        f"max |Δ| = {(x1_new - x1_ref).abs().max().item():.3e}"
    )


def test_fmsolver_euler_independent_of_passing_unused_posterior_args():
    """Passing posterior knobs at their default (no-op) values must not
    perturb the trajectory."""
    torch.manual_seed(7)
    x0 = torch.randn(2, 3, 4)
    fm = FMSolver(velocity_model=_AnalyticV())

    x1_a = fm.sample(x_init=x0, num_step=32, method="euler")
    x1_b = fm.sample(
        x_init=x0,
        num_step=32,
        method="euler",
        time_grid_mode="uniform",
        time_grid_gamma=2.0,
        resample_steps=0,
        resample_t_threshold=0.4,
        valid_length=None,
    )
    assert torch.equal(x1_a, x1_b)


# --------------------------------------------------------------------------- #
# 2. Time grid                                                                #
# --------------------------------------------------------------------------- #


def test_build_time_grid_uniform():
    g = build_time_grid(10, "uniform", 2.0, device="cpu")
    assert g.shape == (11,)
    assert torch.allclose(g, torch.linspace(0.0, 1.0, 11))


def test_build_time_grid_geometric_monotone_and_boundaries():
    g = build_time_grid(10, "geometric", 2.0, device="cpu")
    assert g.shape == (11,)
    assert g[0].item() == 0.0
    assert g[-1].item() == 1.0
    assert torch.all(g[1:] >= g[:-1])  # monotone non-decreasing
    # γ=2 concentrates points near 0 → first half should cover < half the range.
    assert g[5].item() < 0.5


def test_build_time_grid_unknown_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        build_time_grid(10, "weird", 1.0, device="cpu")


# --------------------------------------------------------------------------- #
# 3. RePaint-style inner refinement                                           #
# --------------------------------------------------------------------------- #


class _CountingV(ModelWrapper):
    """Velocity field that counts how many times it was called."""

    def __init__(self):
        super().__init__(model=torch.nn.Identity())
        self.calls = 0

    def forward(self, x, t, **_extras):
        self.calls += 1
        return torch.zeros_like(x)


def test_resample_inner_calls_count_matches_threshold():
    """With K=2 and τ=0.4, inner refinement runs on outer steps with
    ``t < 0.4``. NFE = num_step + (#steps with t<τ) * K."""
    num_step = 10
    K = 2
    tau = 0.4

    v = _CountingV()
    fm = FMSolver(velocity_model=v)
    x0 = torch.zeros(1, 2, 3)

    fm.sample(
        x_init=x0,
        num_step=num_step,
        method="euler",
        resample_steps=K,
        resample_t_threshold=tau,
    )

    # Uniform grid: t_i = i/num_step. Steps with t_i < 0.4 → i ∈ {0,1,2,3} → 4 steps.
    early_steps = sum(1 for i in range(num_step) if i / num_step < tau)
    expected_calls = num_step + early_steps * K
    assert v.calls == expected_calls, (
        f"expected {expected_calls} velocity calls "
        f"({num_step} outer + {early_steps}×{K} inner refinement), got {v.calls}"
    )


def test_resample_zero_pads_invalid_positions():
    """When ``valid_length`` is given, padded positions are kept zero by the
    inner refinement loop (the outer Euler step does NOT zero-pad — that
    happens at the call site in ``SmartMeterFMModel.sample``).

    Use a velocity field with ``v = -x`` so a zero input stays zero across
    both inner refinement and outer Euler updates.
    """
    num_step = 4
    K = 2
    tau = 1.0  # all outer steps trigger inner refinement

    class _LinearV(ModelWrapper):
        def __init__(self):
            super().__init__(model=torch.nn.Identity())

        def forward(self, x, t, **_extras):
            return -x

    fm = FMSolver(velocity_model=_LinearV())
    # B=2, L=4 patches of D=3 channels.
    torch.manual_seed(0)
    # Start from a state where padded regions are already zero — the
    # inner refinement (which adds Gaussian noise) must keep them zero.
    x0 = torch.randn(2, 4, 3)
    x0[0, 2:, :] = 0  # valid_folded[0] = 6 // 3 = 2
    x0[1, 3:, :] = 0  # valid_folded[1] = 9 // 3 = 3
    valid_length = torch.tensor([6, 9])  # full unfolded units

    out = fm.sample(
        x_init=x0,
        num_step=num_step,
        method="euler",
        resample_steps=K,
        resample_t_threshold=tau,
        valid_length=valid_length,
    )

    # Padded patches must still be zero (v=-x preserves zeros at every step,
    # and the inner refinement's fresh-noise re-noising is zeroed back out).
    assert torch.all(out[0, 2:, :] == 0)
    assert torch.all(out[1, 3:, :] == 0)
    # ...but the valid region must have moved away from x0.
    assert (out[0, :2, :] - x0[0, :2, :]).abs().sum().item() > 0
    assert (out[1, :3, :] - x0[1, :3, :]).abs().sum().item() > 0


# --------------------------------------------------------------------------- #
# 4. AutocastForwardWrapper                                                   #
# --------------------------------------------------------------------------- #


def test_autocast_wrapper_disabled_is_passthrough():
    """When ``enabled=False`` the wrapper must not alter the output."""

    class _FpModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

        def forward(self, x, t, **_extras):
            return self.lin(x)

    torch.manual_seed(0)
    inner = _FpModel()
    wrapped = AutocastForwardWrapper(
        inner, device_type="cpu", dtype=torch.bfloat16, enabled=False
    )
    x = torch.randn(2, 8)
    t = torch.tensor(0.3)

    y_ref = inner(x, t)
    y_wrapped = wrapped(x, t)
    assert torch.equal(y_ref, y_wrapped)
    assert y_wrapped.dtype == torch.float32


def test_autocast_wrapper_returns_fp32_under_bf16():
    """Even when matmuls run in bf16, the wrapper output must be fp32 so
    every consumer above (path conversions, CFG blend, projection, the
    integrator's Euler step) sees fp32 tensors."""

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA not available — bf16 autocast smoke test skipped.")

    class _FpModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 16)

        def forward(self, x, t, **_extras):
            return self.lin(x)

    torch.manual_seed(0)
    inner = _FpModel().cuda()
    wrapped = AutocastForwardWrapper(
        inner, device_type="cuda", dtype=torch.bfloat16, enabled=True
    )
    x = torch.randn(2, 16, device="cuda")
    t = torch.tensor(0.3, device="cuda")

    y = wrapped(x, t)
    assert y.dtype == torch.float32
    assert y.shape == (2, 16)
