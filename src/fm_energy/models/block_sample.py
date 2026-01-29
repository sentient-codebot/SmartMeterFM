from collections.abc import Sequence

import torch
from einops import rearrange
from flow_matching.path import AffineProbPath


class _BlockResampleSolve:
    """Iterator for Block Resampling.

    Resample `x_0` to construct N variants of `x_t` for every block of B time steps.

    NOTE: uses its own "solver".
    NOTE: assumes Gaussian source distribution.

    Args:
        x_0: torch.Tensor
            The initial state.
        velocity_model: torch.nn.Module
            The model to predict the velocity.
        path: AffineProbPath
            The path to use for sampling.
        num_steps: int
            The number of time steps to take.
        score_func: callable - f(x_t, t)
            The function used to select the top ones.
        param_N: int
            The number of variants samples to generate.
        param_B: int
            The number of time steps in each block.

    """

    def __init__(
        self,
        x_0: torch.Tensor,
        velocity_model: torch.nn.Module,
        path: AffineProbPath,
        num_steps: int,
        param_N: int,
        param_B: int,
        score_func: callable,
        time_grid: Sequence[float] | None = None,
        extras: dict | None = None,
    ):
        if time_grid is not None:
            self.time_grid = time_grid
        else:
            self.time_grid = [0.0, 1.0]
        self.all_time_steps = torch.linspace(
            self.time_grid[0], self.time_grid[1], num_steps + 1
        ).to(x_0.device)
        self.score_func = score_func
        self.model = velocity_model
        self.path = path
        self.num_steps = num_steps
        self.param_N = param_N
        self.param_B = param_B
        if self.param_B <= 1:
            raise ValueError(
                "param_B must be greater than 1"
            )  # == 1 => resample every step

        self.extras = extras if extras is not None else {}
        # init state
        self.x_t = x_0
        self.current_step = 0

        if x_0.device != next(velocity_model.parameters()).device:
            raise ValueError("x_0 and velocity_model must be on the same device")

    def get_x_1(self, x_t, u_t, t):
        x_1 = self.path.velocity_to_target(u_t, x_t, t)
        return x_1

    def __len__(self):
        """Return the number of time steps."""
        return self.num_steps

    @torch.no_grad()
    def __next__(self):
        """Resample x_0 to construct N variants of x_t for every block of B time steps."""
        if self.current_step >= self.num_steps:
            raise StopIteration

        # get time step
        use_t = self.all_time_steps[self.current_step]

        # branch: resample/select/step
        if self.current_step % self.param_B == 0:
            # resample
            # d_x_t = d_alpha_t x_1 + d_sigma_t x_0
            # x_t shape (B, *dim)
            u_t = self.model(self.x_t, use_t, **self.extras)
            x_1 = self.get_x_1(self.x_t, u_t, use_t)  # shape (batch, *dim)
            x_1 = rearrange(x_1, "B ... -> () B ...")  # shape (1, B, *dim)
            x_1 = x_1.expand(self.param_N, *x_1.shape[1:])  # shape (N, B, *dim)
            x_0 = torch.randn_like(x_1)  # shape (N, B, *dim)
            # <<< should I do?
            # scheduler_out = self.path.scheduler(use_t)
            # new_x_t = self.x_t + (
            #     scheduler_out.d_alpha_t * x_1
            #     + scheduler_out.d_sigma_t * x_0
            # ) * (1.0 / self.num_steps)  # shape (N, B, *dim)
            # >>>>>>>>>>>>>>>>>>>>>>
            # <<<< or, should I do?
            scheudler_out = self.path.scheduler(
                self.all_time_steps[self.current_step + 1]
            )
            new_x_t = scheudler_out.alpha_t * x_1 + scheudler_out.sigma_t * x_0
            # >>>>>>>>>>>>>>>>>>>>>>

            self.x_t = rearrange(new_x_t, "N B ... -> (N B) ...")
            self.current_step += 1
        elif (self.current_step + 1) % self.param_B == 0:
            # step and select
            u_t = self.model(self.x_t, use_t, **self.extras)
            new_x_t = self.x_t + u_t * (1.0 / self.num_steps)  # shape (NB, *dim)
            score = self.score_func(
                self.get_x_1(self.x_t, u_t, use_t), use_t
            )  # shape (NB, 1)
            score = rearrange(score, "(N B) 1 -> B N", N=self.param_N)
            _topk_sorted, topk_idx = torch.topk(
                score,
                k=1,
                dim=1,
                largest=True,
            )  # idx: shape (B, 1)
            topk_idx = topk_idx.squeeze(1)  # shape (B,)
            new_x_t = rearrange(new_x_t, "(N B) ... -> N B ...", N=self.param_N)

            expand_dims = [1] * len(new_x_t.shape)
            expand_dims[1] = topk_idx.shape[0]
            expanded_topk_idx = topk_idx.view(*expand_dims).expand(
                1, *new_x_t.shape[1:]
            )
            selected_x_t = new_x_t.gather(0, expanded_topk_idx)  # shape (B, *dim)

            self.x_t = selected_x_t.squeeze(0)
            self.current_step += 1
        else:
            # step
            u_t = self.model(self.x_t, use_t, **self.extras)
            new_x_t = self.x_t + u_t * (1.0 / self.num_steps)

            self.x_t = new_x_t  # shape (NB, *dim)
            self.current_step += 1

        return self.x_t


# wrapper that is iterable and returns a solver
class BlockResampleSolver:
    """Iterator for Block Resampling.

    Resample `x_0` to construct N variants of `x_t` for every block of B time steps.

    Args:
        x_0: torch.Tensor
            The initial state.
        velocity_model: torch.nn.Module
            The model to predict the velocity.
        path: AffineProbPath
            The path to use for sampling.
        num_steps: int
            The number of time steps to take.
        score_func: callable - f(x_t, t)
            The function used to select the top ones.
        param_N: int
            The number of variants samples to generate.
        param_B: int
            The number of time steps in each block.

    """

    def __init__(
        self,
        x_0: torch.Tensor,
        velocity_model: torch.nn.Module,
        path: AffineProbPath,
        num_steps: int,
        param_N: int,
        param_B: int,
        score_func: callable,
        time_grid: Sequence[float] | None = None,
        extras: dict | None = None,
    ):
        if time_grid is not None:
            self.time_grid = time_grid
        else:
            self.time_grid = [0.0, 1.0]
        self.all_time_steps = torch.linspace(
            self.time_grid[0], self.time_grid[1], num_steps + 1
        ).to(x_0.device)
        self.score_func = score_func
        self.model = velocity_model
        self.path = path
        self.num_steps = num_steps
        self.param_N = param_N
        self.param_B = param_B

        self.extras = extras if extras is not None else {}
        # init state
        self.x_t = x_0
        self.current_step = 0

        if x_0.device != next(velocity_model.parameters()).device:
            raise ValueError("x_0 and velocity_model must be on the same device")

    def __iter__(self):
        return _BlockResampleSolve(
            x_0=self.x_t,
            velocity_model=self.model,
            path=self.path,
            num_steps=self.num_steps,
            param_N=self.param_N,
            param_B=self.param_B,
            score_func=self.score_func,
            time_grid=self.time_grid,
            extras=self.extras,
        )
