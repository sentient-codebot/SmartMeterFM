"SDE samplers"

import torch
from flow_matching.path import AffineProbPath, scheduler
from flow_matching.path.scheduler import ScheduleTransformedModel, VPScheduler
from flow_matching.utils import ModelWrapper
from torch import Tensor


class SDESampleWrapper(ModelWrapper):
    """Wrapper for SDE sampling.

    can only be used with Euler method.
    used as a velocity predictor.
    """

    def __init__(
        self,
        velocity_predictor: torch.nn.Module,
        path: AffineProbPath,
        step_size: float,
        ode_threshold: float | None = None,
    ):
        given_scheduler = path.scheduler
        if not isinstance(given_scheduler, VPScheduler):
            new_scheduler = VPScheduler(0.1, 20)  # deefault 0.1, 20
            velocity_predictor = ScheduleTransformedModel(
                velocity_predictor,
                given_scheduler,
                new_scheduler,
            )
            path = AffineProbPath(scheduler=new_scheduler)
        super().__init__(model=velocity_predictor)
        self.path = path
        self.step_size = step_size
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.ode_threshold = ode_threshold
        if ode_threshold is not None:
            if ode_threshold < 0 or ode_threshold > 1:
                raise ValueError("ode_threshold must be in [0, 1]")
        else:
            self.ode_threshold = 1.0

    def _get_beta_t(self, t: Tensor) -> Tensor:
        """get beta_sde_t = -1/2 log(alpha_t)"""
        B, b = self.path.scheduler.beta_max, self.path.scheduler.beta_min
        return 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b

    def get_score_coef(self, t: Tensor) -> Tensor:
        """
        u_t = a_t * x_t + b_t * s_t; s_t is score, u_t is velocity.

        Arguments:
            t -- tensor, time

        Returns:
            a_t, b_t -- tensor, time-dependent coefficients

        NOTICE singularity:
            - a_t = d_alpha_t / alpha_t, at t = 0, might be singularity.
            - b_t = -(...) / alpha_t, at t = 0, might be singularity.
        """
        # for singularity
        if isinstance(self.path.scheduler, scheduler.CondOTScheduler):
            t = t.clamp(min=1e-4)
        # for VPScheduler, d_alpha_t > 0, alpha_t ~= 0, > 0.
        sch_out = self.path.scheduler(t)
        alpha_t, sigma_t, d_sigma_t, d_alpha_t = (
            sch_out.alpha_t,
            sch_out.sigma_t,
            sch_out.d_sigma_t,
            sch_out.d_alpha_t,
        )
        b_t = -(d_sigma_t * sigma_t * alpha_t - d_alpha_t * sigma_t**2) / alpha_t
        a_t = d_alpha_t / alpha_t
        return a_t, b_t

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        """predict (noised velocity)
        Conversion formula:
            drift <- velocity
            d(x, t) = 2 u(x, t) - a_t * x

        Arguments:
            x -- x_t
            t -- t

        Returns:
            drift + noise/dT - drift + noise rescaled like velocity
        """
        u_t = self.model(x, t, **extras)
        # get coef
        a_t, b_t = self.get_score_coef(t)
        # get drift
        drift_t = 2 * u_t - a_t * x
        # get noise
        z = torch.randn_like(x)
        noise_t = (2 * b_t / self.step_size) ** 0.5 * z
        return torch.where(
            t > self.ode_threshold,
            u_t,
            drift_t + noise_t,
        )


class AdaptiveSDESampleWrapper(ModelWrapper):
    """Wrapper for Adaptive SDE sampling.

    Sample first using SDE with `step_size` until `t` hits `ode_threshold`.
    Then switch to ODE sampling with `step_size * ode_eval_every`, which practically
    skips computation and maintains the same velocity for `ode_eval_every` steps.
    """

    def __init__(
        self,
        velocity_predictor: torch.nn.Module,
        path: AffineProbPath,
        step_size: float,
        ode_threshold: float | None = None,
        ode_eval_every: int = 0,
    ):
        given_scheduler = path.scheduler
        if not isinstance(given_scheduler, VPScheduler):
            new_scheduler = VPScheduler(0.1, 20)  # default 0.1, 20
            velocity_predictor = ScheduleTransformedModel(
                velocity_predictor,
                given_scheduler,
                new_scheduler,
            )
            path = AffineProbPath(scheduler=new_scheduler)
        super().__init__(model=velocity_predictor)
        self.path = path
        self.step_size = step_size
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.ode_threshold = ode_threshold
        if ode_threshold is not None:
            if ode_threshold < 0 or ode_threshold > 1:
                raise ValueError("ode_threshold must be in [0, 1]")
        else:
            self.ode_threshold = 1.0
        self.ode_eval_every = ode_eval_every
        if ode_eval_every < 0:
            raise ValueError("ode_skip_every must be a positive integer.")
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.int64))

    def beta_schedule(self, t: Tensor) -> Tensor:
        """NOTE: why does this matter? shouldn't this be arbitrary and we would
        have the same marginal p_t?
        """
        # t in [0, 1]
        # return torch.exp(t / 1)
        _B, b = self.path.scheduler.beta_max, self.path.scheduler.beta_min
        return b  # adding Langevin dynamics with constant step size b.
        # return 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b

    def get_score_coef(self, t: Tensor) -> Tensor:
        """
        u_t = a_t * x_t + b_t * s_t; s_t is score, u_t is velocity.

        Arguments:
            t -- tensor, time

        Returns:
            a_t, b_t -- tensor, time-dependent coefficients

        NOTICE singularity:
            - a_t = d_alpha_t / alpha_t, at t = 0, might be singularity.
            - b_t = -(...) / alpha_t, at t = 0, might be singularity.
        """
        # for singularity
        if isinstance(self.path.scheduler, scheduler.CondOTScheduler):
            t = t.clamp(min=1e-4)
        # for VPScheduler, d_alpha_t > 0, alpha_t ~= 0, > 0.
        sch_out = self.path.scheduler(t)
        alpha_t, sigma_t, d_sigma_t, d_alpha_t = (
            sch_out.alpha_t,
            sch_out.sigma_t,
            sch_out.d_sigma_t,
            sch_out.d_alpha_t,
        )
        b_t = -(d_sigma_t * sigma_t * alpha_t - d_alpha_t * sigma_t**2) / alpha_t
        a_t = d_alpha_t / alpha_t
        return a_t, b_t

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        """predict (noised velocity)
        Conversion formula:
            drift <- velocity
            d(x, t) = 2 u(x, t) - a_t * x

        Arguments:
            x -- x_t
            t -- t

        Returns:
            drift + noise/dT - drift + noise rescaled like velocity
        """
        if torch.any(t < self.ode_threshold):
            u_t = self.model(x, t, **extras)
            # get coef
            a_t, b_t = self.get_score_coef(t)
            # get score
            s_t = (u_t - a_t * x) / b_t
            # get drift
            beta_t = self.beta_schedule(t)  # Langevin 1/2 beta_t**2 s_t dt + beta_t dw
            drift_t = u_t + 1 / 2 * beta_t**2 * s_t
            # drift_t = 2 * u_t - a_t * x  # only when 1/2 beta_t**2 == b_t
            # get diffusion
            z = torch.randn_like(x)
            diffusion_coef = beta_t / self.step_size**0.5
            diffusion_t = diffusion_coef * z
            # noise_t = (2 * b_t / self.step_size) ** 0.5 * z
            # pprint(f"{t.item(): .3f}, {u_t.flatten(start_dim=1).norm(dim=1).mean(): .4f}")
            # pprint(f"{beta_t.item(): .4f}, {(2*b_t)**0.5: .4f}")
            return torch.where(
                t > self.ode_threshold,
                u_t,
                drift_t + diffusion_t,
            )
        else:
            if self.step_counter.item() % self.ode_eval_every == 0:
                self.step_counter += 1
                u_t = self.model(x, t, **extras)
                self.register_buffer("last_u_t", u_t.detach())
                return u_t
            else:
                self.step_counter += 1
                return self.last_u_t

        # return torch.where(
        #     t > self.ode_threshold,
        #     u_t,
        #     drift_t + noise_t,
        # )
