"""scripts for constructing, training, and sampling from flow models."""

import enum
from collections.abc import Callable

import pytorch_lightning as pl
import torch
from einops import rearrange
from ema_pytorch import EMA
from flow_matching.path import AffineProbPath, scheduler
from flow_matching.path.scheduler import ScheduleTransformedModel, VPScheduler
from flow_matching.utils import ModelWrapper
from jaxtyping import Float
from torch import Tensor

from ..utils.configuration import FMConfig, SampleConfig, TrainConfig, TransformerConfig
from .embedders import get_embedder
from .measurement import LinearOperator, Noise, NonLinearOperator
from .nn_components import DenoisingTransformer, get_start_pos


class PredictionType(str, enum.Enum):
    VELOCITY = "velocity"
    X0 = "x0"
    X1 = "x1"


class X1ToVelocityModelWrapper(ModelWrapper):
    def __init__(self, denoiser: torch.nn.Module, path: AffineProbPath):
        super().__init__(model=denoiser)
        self.path = path

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        x_1_prediction = super().forward(x, t, **extras)
        return self.path.target_to_velocity(x_1=x_1_prediction, x_t=x, t=t)


class X0ToVelocityModelWrapper(ModelWrapper):
    def __init__(self, noise_predictor: torch.nn.Module, path: AffineProbPath):
        super().__init__(model=noise_predictor)
        self.path = path

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        noise_prediction = super().forward(x, t, **extras)
        return self.path.epsilon_to_velocity(epsilon=noise_prediction, x_t=x, t=t)


class ValidLengthModelWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module, valid_length: Tensor, *args, **kwargs):
        super().__init__(model=model)
        self.valid_length = valid_length
        if valid_length.ndim != 1:
            raise ValueError("valid_length must be 1D tensor")

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return self.model(x, t, valid_length=self.valid_length, **extras)


class ReverseVelocityModelWrapper(ModelWrapper):
    """reverse the speed for the sake of forward."""

    def __init__(self, velocity_predictor: torch.nn.Module):
        super().__init__(model=velocity_predictor)

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return -self.model(x, t, **extras)  # reverse the speed


class PositionAlignmentModelWrapper(ModelWrapper):
    """Align the first day of the week and month length.

    Args:
        model (nn.Module): the model to wrap.
        path (AffineProbPath). used to calculate x_1
        valid_length (Tensor): the valid length of the input.
        start_pos (Tensor | int): the start position of the input.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        path: AffineProbPath,
        valid_length: Tensor,
        start_pos: Tensor | int,
    ):
        super().__init__(model=model)
        self.path = path
        self.valid_length = valid_length
        self.start_pos = start_pos
        if valid_length.ndim != 1:
            raise ValueError("valid_length must be 1D tensor")
        if isinstance(start_pos, Tensor):
            if start_pos.ndim != 1:
                raise ValueError("start_pos must be 1D tensor")
            if start_pos.shape[0] != valid_length.shape[0]:
                raise ValueError(
                    "start_pos and valid_length\
                    must have the same shape"
                )
        elif not isinstance(start_pos, int):
            raise TypeError("start_pos must be int or 1D tensor")

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return self.model(
            x=x, t=t, start_pos=self.start_pos, valid_length=self.valid_length, **extras
        )


class PosteriorMethod(str, enum.Enum):
    DPS = "dps"
    PROJECT = "project"
    GRADIENT_DESCENT = "gd"


class PosteriorVelocityModelWrapper(ModelWrapper):
    """Abstract wrapper for posterior velocity models.

    Arguments:
        velocity_predictor: a velocity predictor.
        path: AffineProbPath object. used to calculate x_1

    Need to implement:
        - measurement_loglikelihood:
        input:
            - x: x_t
            - t: t
            - velocity_prediction: u_t, the velocity prediction
        output:
            - loglikelihood of the measurement

    """

    def __init__(
        self,
        velocity_predictor: torch.nn.Module,
        path: AffineProbPath,
        operator: LinearOperator | NonLinearOperator,
        noise_model: Noise,
        measurement_y: Tensor,
        valid_length: int | None,
        scale: float = 1.0,
        num_sampling: int = 1,
        project: bool = False,
        clamp_t: bool = True,
        method: PosteriorMethod = PosteriorMethod.DPS,
    ):
        training_scheduler = path.scheduler
        if not isinstance(training_scheduler, VPScheduler):
            sampling_scheduler = VPScheduler(0.1, 20)  # default beta [0.1, 20]
            velocity_predictor = ScheduleTransformedModel(
                velocity_model=velocity_predictor,
                original_scheduler=training_scheduler,
                new_scheduler=sampling_scheduler,
            )
            path = AffineProbPath(scheduler=sampling_scheduler)
        else:
            path = AffineProbPath(scheduler=training_scheduler)
        super().__init__(model=velocity_predictor)
        self.path = path
        self.operator = operator
        self.noise_model = noise_model
        self.measurement_y = measurement_y.detach().clone()
        self.scale = scale
        if scale < 0:
            raise ValueError("scale must be non-negative")
        self.num_sampling = num_sampling
        if num_sampling < 1:
            raise ValueError("num_sampling must be no less than 1")
        self.project = project
        self.clamp_t = clamp_t
        self.valid_length = valid_length
        self.method = method
        if self.method == PosteriorMethod.PROJECT and not hasattr(
            self.operator, "project"
        ):
            self.method = PosteriorMethod.GD
        self.gd_steps = 10
        self.gd_stepsize = 1.0

    def predict_x1(
        self, x_t: Tensor, t: Tensor, velocity_prediction: Tensor
    ) -> Float[Tensor, "batch *"]:
        """Predict x_1 from x_t and velocity prediction."""
        if t.shape != x_t.shape:
            for _i in range(x_t.dim() - t.dim()):
                t = t.unsqueeze(-1)

        # Calculate the x1_prediction
        x1 = self.path.velocity_to_target(velocity_prediction, x_t=x_t, t=t)

        return x1

    def predict_ut(
        self, x_t: Tensor, t: Tensor, target_prediction: Tensor
    ) -> Float[Tensor, "batch *"]:
        """Predict u_t from x_t and target prediction."""
        if t.shape != x_t.shape:
            for _i in range(x_t.dim() - t.dim()):
                t = t.unsqueeze(-1)

        # Calculate the u_t_prediction
        u_t = self.path.target_to_velocity(target_prediction, x_t=x_t, t=t)

        return u_t

    def measurement_loglikelihood(
        self,
        predicted_x1: Tensor,
        measurement_y: Tensor,
    ) -> Float[Tensor, ""]:
        r"""Calculate $log p(y|x1)$"""
        if predicted_x1.dim() == 3:
            predicted_x1 = rearrange(
                predicted_x1,
                "batch sequence channel -> batch (sequence channel)",
            )
            # truncate if necessary
            if self.valid_length is not None:
                predicted_x1 = predicted_x1[:, : self.valid_length]
        return self.noise_model.log_likelihood(
            pseudo_measurement=self.operator.forward(predicted_x1),
            measurement=measurement_y,
        ).sum()  # scalar

    def disable_model_grad(self) -> None:
        """Disable model grad."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        momentum_buffer: "MomentumBuffer | None" = None,
        **extras,
    ) -> Tensor:
        if self.method == PosteriorMethod.DPS:
            x.requires_grad_(True)
            with torch.enable_grad():
                velocity_prior = self.model(x, t, **extras)
                x1_pred = self.predict_x1(x, t, velocity_prior)
                average_loglikelihood = 0
                for _ in range(self.num_sampling):
                    # NOTE: even when num_sampling == 1, perturbation is added.
                    if self.num_sampling > 1:
                        x1_pred_perturb = x1_pred + torch.randn_like(x1_pred) * 0.05
                    else:
                        x1_pred_perturb = x1_pred
                    loglikelihood = self.measurement_loglikelihood(
                        predicted_x1=x1_pred_perturb,
                        measurement_y=self.measurement_y,
                    )
                    average_loglikelihood += loglikelihood / self.num_sampling

                velocity_shift = torch.autograd.grad(
                    outputs=average_loglikelihood,  # loglikelihood
                    inputs=x,  # x_t
                )[0]
            sch_out = self.path.scheduler(t.clamp(min=1e-3, max=1 - 1e-3))
            alpha_t, sigma_t, d_sigma_t, d_alpha_t = (
                sch_out.alpha_t,
                sch_out.sigma_t,
                sch_out.d_sigma_t,
                sch_out.d_alpha_t,
            )
            b_t = (
                -1 / alpha_t * (d_sigma_t * sigma_t * alpha_t - d_alpha_t * sigma_t**2)
            )
            # b_t.clamp(min=-10, max=10)
            # for CondOTPath, b_t = -1 + 1/t, (+inf -> 0)

            w = self.scale  # 1.0 works, 1.5 also works; also depends on sigma
            # velocity_posterior = velocity_prior + w * b_t * velocity_shift
            velocity_posterior = apg_forward(
                velocity_prior + b_t * velocity_shift,
                velocity_prior,
                guidance_scale=w,
                momentum_buffer=momentum_buffer,
            )

            # project if necessary
            with torch.no_grad():
                if self.project and hasattr(self.operator, "project"):
                    _x1_pred = self.predict_x1(x, t, velocity_posterior)
                    # pre-project
                    B, L, D = x.shape
                    if self.valid_length is not None:
                        valid_mask = torch.zeros(
                            (B, L * D),
                            device=x.device,
                            dtype=torch.bool,
                        )
                        valid_mask[:, : self.valid_length] = True
                        _x1_valid = _x1_pred.flatten(1)[
                            :, : self.valid_length
                        ]  # [B, <= L*D]
                    else:
                        _x1_valid = _x1_pred.flatten(1)  # [B, L*D]
                    # project
                    _x1_projected = self.operator.project(_x1_valid, self.measurement_y)
                    # post-project
                    if _x1_projected.shape[1] < L * D:
                        _zeros = _x1_projected.new_zeros(
                            (_x1_projected.shape[0], L * D - _x1_projected.shape[1]),
                        )
                        x1_projected = torch.cat([_x1_projected, _zeros], dim=1).view(
                            x.shape
                        )  # [B, L*D]
                    else:
                        x1_projected = _x1_projected.view(x.shape)
                    velocity_posterior = self.predict_ut(
                        x, t.clamp(max=1.0 - 1e-4), x1_projected
                    )

        elif self.method == PosteriorMethod.PROJECT:
            if x.ndim == 2:
                B, L, D = *x.shape, 1
            elif x.ndim == 3:
                B, L, D = x.shape
            else:
                raise ValueError(f"Invalid input shape: {x.shape}")
            velocity_prior = self.model(x, t, **extras)
            x1_pred = self.predict_x1(x, t, velocity_prior)
            if self.valid_length is not None:
                valid_mask = torch.zeros(
                    (B, L * D),
                    device=x.device,
                    dtype=torch.bool,
                )
                valid_mask[:, : self.valid_length] = True
                x1_valid = x1_pred.flatten(1)[:, : self.valid_length]  # [B, <= L*D]
            else:
                x1_valid = x1_pred.flatten(1)  # [B, L*D]
            _x1_projected = self.operator.project(
                x1_valid, self.measurement_y
            )  # [B, <= L*D]
            if _x1_projected.shape[1] < L * D:
                _zeros = _x1_projected.new_zeros(
                    (_x1_projected.shape[0], L * D - _x1_projected.shape[1]),
                )
                x1_projected = torch.cat([_x1_projected, _zeros], dim=1).view(
                    x.shape
                )  # [B, L*D]
            else:
                x1_projected = _x1_projected.view(x.shape)  # [B, L*D]
            velocity_guided = self.predict_ut(x, t.clamp(max=1.0 - 1e-4), x1_projected)
            velocity_posterior = apg_forward(
                velocity_guided,
                velocity_prior,
                guidance_scale=self.scale,
                momentum_buffer=momentum_buffer,
            )
        else:  # PosteriorMethod.GRADIENT_DESCENT
            optimizer = MiniNesterovSGD(lr=self.gd_stepsize)
            velocity_prior = self.model(x, t, **extras)
            x1_pred_ori = self.predict_x1(x, t, velocity_prior)
            x1_pred = x1_pred_ori.clone()
            idx_gd_step = 0
            while idx_gd_step < self.gd_steps:
                with torch.enable_grad():
                    full_length = (
                        x1_pred.shape[1] * x1_pred.shape[2]
                        if x1_pred.dim() == 3
                        else x1_pred.shape[1]
                    )
                    if (
                        self.valid_length is not None
                        and self.valid_length < full_length
                    ):
                        _valid = x1_pred.flatten(1)[
                            :, : self.valid_length
                        ].requires_grad_(True)
                        _non_valid = x1_pred.flatten(1)[
                            :, self.valid_length :
                        ].requires_grad_(False)
                        x1_pred = torch.cat([_valid, _non_valid], dim=1).view(
                            x1_pred.shape
                        )
                    else:
                        x1_pred = x1_pred.requires_grad_(True)
                    llh = self.measurement_loglikelihood(
                        x1_pred,
                        self.measurement_y,
                    )
                    # reg = torch.sum((self.path.target_to_epsilon(
                    #     x_1=x1_pred, x_t=x, t=t
                    # ).flatten(1).norm(dim=-1, p=2) - 1.0)**2)  # variance reg
                    reg_2 = torch.sum((x1_pred - x1_pred_ori) ** 2)  # ~= projection
                    opt_obj = llh - 0.01 * reg_2  # loglikelihood + regularization
                g = torch.autograd.grad(
                    outputs=opt_obj,  # loglikelihood
                    inputs=x1_pred,  # x1_pred
                )[0]
                x1_pred = optimizer.step(x1_pred.detach(), g)
                # x1_pred = x1_pred.detach() + self.gd_stepsize * g
                idx_gd_step += 1
            velocity_guided = self.predict_ut(x, t.clamp(max=1.0 - 1e-4), x1_pred)
            velocity_posterior = apg_forward(
                velocity_guided,
                velocity_prior,
                guidance_scale=self.scale,
                momentum_buffer=momentum_buffer,
            )

        return velocity_posterior.detach()  # probably detach not necessary


class StartPosWrapper(ModelWrapper):
    """Wrapper that passes a fixed (int/batched tensor) `start_pos` to the model.

    Args:
        model: the model to wrap.
        path: the probability path. not used but kept for consistency.
        start_pos: the start position to pass to the model.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        path: AffineProbPath,
        start_pos: int | Tensor,
    ):
        self.model = model
        self.path = path
        self.start_pos = start_pos

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        if isinstance(self.start_pos, Tensor):
            self.start_pos = self.start_pos.to(x.device)
        out = self.model(
            x=x,
            t=t,
            start_pos=self.start_pos,
            **extras,
        )
        return out


class ConditionedVelocityModelWrapper(torch.nn.Module):
    """Wrapper around velocity model to inject conditions during inference."""

    def __init__(self, velocity_model, dict_labels, dict_emb_extras, cfg_scale=1.0):
        super().__init__()
        self.velocity_model = velocity_model
        self.dict_labels = dict_labels
        self.dict_emb_extras = dict_emb_extras
        self.cfg_scale = cfg_scale

    def forward(self, x, t, **kwargs):
        """Forward pass with classifier-free guidance."""
        # If cfg_scale is 1.0 or no conditions, just use the regular model
        if self.cfg_scale == 1.0 or self.dict_labels is None:
            return self.velocity_model(
                x, t, c=self.dict_labels, dict_emb_extras=self.dict_emb_extras, **kwargs
            )

        batch_size = x.shape[0]

        # Handle scalar t (convert to batch)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        # Duplicate inputs for conditional and unconditional passes
        x_doubled = torch.cat([x, x], dim=0)
        t_doubled = torch.cat([t, t], dim=0)

        # Repeat conditions
        c_doubled = {k: torch.cat([v, v], dim=0) for k, v in self.dict_labels.items()}
        # Create force_drop_ids: keep conditions for first half, drop for second half
        force_drop_ids_doubled = torch.cat(
            [
                torch.zeros(
                    batch_size, dtype=torch.long, device=x.device
                ),  # keep conditions
                torch.ones(
                    batch_size, dtype=torch.long, device=x.device
                ),  # drop conditions
            ],
            dim=0,
        )

        def expand_extra_dict(in_dict):
            for k, v in in_dict.items():
                if k == "force_drop_ids":
                    in_dict[k] = force_drop_ids_doubled
                    continue
                # in case of other tensors, we need to double them
                if isinstance(v, torch.Tensor):
                    in_dict[k] = torch.cat([v, v], dim=0)
                elif isinstance(v, dict):
                    in_dict[k] = expand_extra_dict(v)

            return in_dict

        extra_doubled = expand_extra_dict(self.dict_emb_extras)

        # Forward pass with doubled batch
        v_doubled = self.velocity_model(
            x_doubled, t_doubled, c=c_doubled, dict_emb_extras=extra_doubled, **kwargs
        )

        # Split results and apply guidance
        v_cond, v_null = v_doubled.chunk(2, dim=0)
        guided_velocity = (1 - self.cfg_scale) * v_null + self.cfg_scale * v_cond

        return guided_velocity


class FlowModelPL(pl.LightningModule):
    def __init__(
        self,
        flow_config: FMConfig,
        model_config: TransformerConfig,
        train_config: TrainConfig,
        sample_config: SampleConfig,
        num_in_channel: int,
        label_embedder_name: str | None = None,
        label_embedder_args: dict | None = None,
        context_embedder_name: str | None = None,
        context_embedder_args: dict | None = None,
        metrics_factory: Callable | None = None,
        create_mask: bool = False,
    ):
        super().__init__()
        label_embedder = None
        if label_embedder_name is not None:
            label_embedder_args = label_embedder_args or {}
            label_embedder = get_embedder(label_embedder_name, **label_embedder_args)
        context_embedder = None
        if context_embedder_name is not None:
            context_embedder_args = context_embedder_args or {}
            context_embedder = get_embedder(
                context_embedder_name, **context_embedder_args
            )
        self.model = DenoisingTransformer(
            dim_base=model_config.dim_base,
            num_in_channel=num_in_channel,
            dim_out=num_in_channel,
            num_attn_head=model_config.num_attn_head,
            num_decoder_layer=model_config.num_decoder_layer,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
            label_embedder=label_embedder,
            context_embedder=context_embedder,
        )
        self.num_in_channel = num_in_channel
        self.ema = EMA(
            self.model,
            beta=train_config.ema_decay,
            update_after_step=100,
            update_every=train_config.ema_update_every,
        )
        self.prediction_type = PredictionType(flow_config.prediction_type)
        self.path = AffineProbPath(scheduler=scheduler.CondOTScheduler())
        self.model_config = model_config
        self.train_config = train_config
        self.sample_config = sample_config
        self.metrics = metrics_factory() if metrics_factory is not None else None
        self.create_mask = create_mask

        self.save_hyperparameters()

    def compile(self, *args, **kwargs) -> None:
        "Compile the model and EMA model. all arguments are passed to their `compile`."
        self.model.compile(*args, **kwargs)
        self.ema.compile(*args, **kwargs)

    def get_ema_velocity_model(self):
        if self.prediction_type == PredictionType.VELOCITY:
            return self.ema.ema_model
        elif self.prediction_type == PredictionType.X0:
            return X0ToVelocityModelWrapper(
                noise_predictor=self.ema.ema_model,
                path=self.path,
            )
        elif self.prediction_type == PredictionType.X1:
            return X1ToVelocityModelWrapper(
                denoiser=self.ema.ema_model,
                path=self.path,
            )
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.lr,
            betas=self.train_config.adam_betas,
        )
        return optimizer

    @staticmethod
    def _convert_offset_month_length(month_length: int | Tensor, offset: int):
        "return `(month_length+offset)*96"
        return (month_length + offset) * 96

    @staticmethod
    def _create_loss_mask(valid_length: Tensor, full_length: int) -> Tensor:
        """Create a mask for the loss function."""
        valid_length = valid_length.long()
        batch_size = valid_length.shape[0]
        loss_mask = torch.zeros(batch_size, full_length)
        for i in range(batch_size):
            loss_mask[i, : valid_length[i]] = 1.0
        return loss_mask

    def training_step(
        self,
        batch: tuple[Tensor, dict[str, Tensor]],
        batch_idx: int,
    ):
        self.train()
        profile, condition = batch
        t = torch.rand(profile.shape[0]).to(profile.device)  # t in [0, 1]
        x_0 = torch.randn_like(profile)
        sample = self.path.sample(x_0=x_0, t=t, x_1=profile)
        target = {
            PredictionType.VELOCITY: sample.dx_t,
            PredictionType.X0: sample.x_0,
            PredictionType.X1: sample.x_1,
        }[self.prediction_type]

        # get start position
        if "first_day_of_week" in condition:
            start_pos = get_start_pos(
                condition["first_day_of_week"].squeeze(1),  # [batch]
                steps_per_day=96,
            )  # [batch]
        else:
            start_pos = 0
        # valid length mas
        if self.create_mask:
            valid_length = self._convert_offset_month_length(
                condition["month_length"], 28
            ).squeeze(1)
            loss_mask = self._create_loss_mask(
                valid_length=valid_length,
                full_length=profile.shape[1] * profile.shape[2],
            )
            loss_mask = loss_mask.to(profile.device)
            loss_mask = rearrange(
                loss_mask,
                "batch (sequence channel) -> batch sequence channel",
                sequence=profile.shape[1],
            )
            model_out = self.model(
                x=sample.x_t,
                t=t,
                start_pos=start_pos,
                c=condition,
                valid_length=valid_length,
            )
            cm_loss = (
                torch.pow((model_out - target) * loss_mask, 2).sum() / loss_mask.sum()
            )  # CM loss
        else:
            cm_loss = torch.pow(
                self.model(x=sample.x_t, t=t, start_pos=start_pos, c=condition)
                - target,
                2,
            ).mean()  # CM loss
        self.log(
            "Train/loss",
            cm_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return cm_loss

    def on_train_batch_end(self, *args, **kwargs):
        self.ema.update()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        self.eval()
        profile, condition = batch
        # val loss
        t = torch.rand(profile.shape[0]).to(profile.device)  # t in [0, 1]
        x_0 = torch.randn_like(profile)
        sample = self.path.sample(x_0=x_0, t=t, x_1=profile)
        target = {
            PredictionType.VELOCITY: sample.dx_t,
            PredictionType.X0: sample.x_0,
            PredictionType.X1: sample.x_1,
        }[self.prediction_type]
        if self.create_mask:
            valid_length = self._convert_offset_month_length(
                condition["month_length"], 28
            ).squeeze(1)
            loss_mask = self._create_loss_mask(
                valid_length=valid_length,
                full_length=profile.shape[1] * profile.shape[2],
            )
            loss_mask = loss_mask.to(profile.device)
            loss_mask = rearrange(
                loss_mask,
                "batch (sequence channel) -> batch sequence channel",
                sequence=profile.shape[1],
            )
            model_out = self.model(
                x=sample.x_t, t=t, c=condition, valid_length=valid_length
            )
            cm_loss = (
                torch.pow((model_out - target) * loss_mask, 2).sum() / loss_mask.sum()
            )  # CM loss
        else:
            cm_loss = torch.pow(
                self.model(x=sample.x_t, t=t, c=condition) - target, 2
            ).mean()  # CM loss
        self.log(
            "Validation/loss",
            cm_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return None


# mini Adam for GD
class MiniAdam:
    def __init__(self, lr: float = 0.1, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Perform a single step of mini Adam.

        Args:
            x (torch.Tensor): the parameters to update.
            g (torch.Tensor): the gradient of the parameters.

        Returns:
            torch.Tensor: the updated parameters.

        """
        if self.m is None:
            self.m = torch.zeros_like(x)  # Initialize first moment
            self.v = torch.zeros_like(x)  # Initialize second moment

        self.t += 1
        beta1, beta2 = self.betas

        # Update biased first moment estimate
        self.m = beta1 * self.m + (1 - beta1) * g
        # Update biased second raw moment estimate
        self.v = beta2 * self.v + (1 - beta2) * (g**2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - beta1**self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - beta2**self.t)

        # Update parameters
        x = x + self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return x


# mini SGD with Nesterov momentum
class MiniNesterovSGD:
    def __init__(self, lr: float = 0.1, momentum: float = 0.5):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Perform a single step of mini Nesterov SGD.

        Args:
            x (torch.Tensor): the parameters to update.
            g (torch.Tensor): the gradient of the parameters.

        Returns:
            torch.Tensor: the updated parameters.

        """
        if self.v is None:
            self.v = torch.zeros_like(x)

        # Update velocity with Nesterov momentum
        self.v = self.momentum * self.v + g

        # Update parameters
        x = x + self.lr * (g + self.momentum * self.v)

        return x


# APG tools
class MomentumBuffer:
    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,  # [B, L, D]
    v1: torch.Tensor,  # [B, L, D]
    dims=(-2,),
):
    dtype = v0.dtype
    device = v0.device
    if device.type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()

    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype).to(device), v0_orthogonal.to(dtype).to(device)


def apg_forward(
    pred_cond: torch.Tensor,  # [B, L, D]
    pred_uncond: torch.Tensor,  # [B, L, D]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims=(-2,),
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided
