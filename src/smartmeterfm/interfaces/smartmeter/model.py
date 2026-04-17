"""Unified sampling interface for SmartMeterFM Flow Matching checkpoints.

Mirrors ``AllianderModel`` from the FM-Energy reference project but trimmed to
the SmartMeterFM use case:

* single-device only (no multi-GPU pool);
* ``condition`` is always required (no unconditional branch);
* posterior sampling is restricted to the ``PROJECT`` method, which uses
  ``operator.project`` to replace observed components in ``x_1_pred`` and
  converts back to velocity via the probability path.

Wrapper composition (outermost last, matching how ``ODESolver`` sees it):

    base velocity model  (EMA + prediction-type -> velocity)
        |
    ConditionedVelocityModelWrapper          (classifier-free guidance)
        |
    PositionAlignmentModelWrapper            (only when condition has
                                              month_length + first_day_of_week)
        |
    PosteriorVelocityModelWrapper            (only when posterior_config is set;
                                              method = PROJECT)
        |
    ODESolver.sample(...)
"""

from __future__ import annotations

import torch
from einops import rearrange
from flow_matching.solver import ODESolver
from torch import Tensor

from smartmeterfm.models import flow as flow_module
from smartmeterfm.models.flow import (
    ConditionedVelocityModelWrapper,
    FlowModelPL,
    PositionAlignmentModelWrapper,
    PosteriorMethod,
    PosteriorVelocityModelWrapper,
)
from smartmeterfm.models.measurement import (
    CompositeOperator,
    get_noise,
    get_operator,
)
from smartmeterfm.models.nn_components import get_start_pos
from smartmeterfm.utils.configuration import (
    FMConfig,
    TrainConfig,
    TransformerConfig,
)
from smartmeterfm.utils.configuration import (
    SampleConfig as _ModelSampleConfig,
)

from ..base.config import ModelConfig, SampleConfig
from ..base.model import ModelInterface
from ..base.posterior import (
    CompositeOperatorConfig,
    OperatorConfig,
    PosteriorSampleConfig,
)


class SmartMeterFMModel(ModelInterface):
    """Concrete ``ModelInterface`` wrapping a ``FlowModelPL`` checkpoint."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.devices = model_config.devices
        if len(self.devices) != 1:
            raise ValueError(
                f"{type(self).__name__} is single-device; got {self.devices}"
            )
        self.device = self.devices[0]

        self.pl_model: FlowModelPL = self._load_model()
        self.path = self.pl_model.path
        # `num_in_channel` is the patch size (innermost dim, stored as channels).
        self.patch_size = self.pl_model.num_in_channel
        # Folded sequence length (number of patches).
        self.num_patches = self.pl_model.model_config.seq_length
        self.seq_length = self.num_patches * self.patch_size
        self.steps_per_day = self.pl_model.steps_per_day
        self.create_mask = self.pl_model.create_mask

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str, device: str | torch.device = "cuda"
    ) -> SmartMeterFMModel:
        """Convenience constructor for showcase scripts."""
        return cls(ModelConfig(model_path=str(checkpoint), devices=device))

    def _load_model(self) -> FlowModelPL:
        path = self.model_config.model_path
        if path.startswith("s3://"):
            raise NotImplementedError(
                "S3 checkpoint loading is not supported in SmartMeterFMModel."
            )
        with torch.serialization.safe_globals(
            [FMConfig, TransformerConfig, TrainConfig, _ModelSampleConfig]
        ):
            return FlowModelPL.load_from_checkpoint(
                path,
                map_location=self.device,
                weights_only=False,
            )

    def _get_base_velocity_model(self, use_ema: bool) -> torch.nn.Module:
        if use_ema:
            base = self.pl_model.get_ema_velocity_model()
        else:
            inner = self.pl_model.model
            inner.eval()
            if self.pl_model.prediction_type.value == "velocity":
                base = inner
            elif self.pl_model.prediction_type.value == "x0":
                base = flow_module.X0ToVelocityModelWrapper(
                    noise_predictor=inner, path=self.path
                )
            else:
                base = flow_module.X1ToVelocityModelWrapper(
                    denoiser=inner, path=self.path
                )
        base.eval()
        return base.to(self.device)

    def _wrap_conditional(
        self,
        v_model: torch.nn.Module,
        condition: dict[str, Tensor],
        cfg_scale: float,
    ) -> torch.nn.Module:
        return ConditionedVelocityModelWrapper(
            velocity_model=v_model,
            dict_labels=condition,
            dict_emb_extras={},
            cfg_scale=cfg_scale,
        )

    def _wrap_position_alignment(
        self,
        v_model: torch.nn.Module,
        condition: dict[str, Tensor],
    ) -> tuple[torch.nn.Module, Tensor | None]:
        """Attach per-sample ``valid_length`` + ``start_pos`` when applicable.

        Returns the (possibly wrapped) model and the valid-length tensor in
        FULL (unfolded) units so the caller can reuse it for x_0 padding.
        """
        if "month_length" not in condition or "first_day_of_week" not in condition:
            return v_model, None

        month_length = condition["month_length"].squeeze(1).to(self.device)
        valid_length = (month_length + 28) * self.steps_per_day
        start_pos = get_start_pos(
            first_day_of_week=condition["first_day_of_week"].squeeze(1).to(self.device),
            steps_per_day=self.steps_per_day,
        )
        wrapped = PositionAlignmentModelWrapper(
            model=v_model,
            path=self.path,
            valid_length=valid_length,
            start_pos=start_pos,
        )
        return wrapped, valid_length

    def _wrap_posterior(
        self,
        v_model: torch.nn.Module,
        posterior_config: PosteriorSampleConfig,
        measurement_y: Tensor,
    ) -> torch.nn.Module:
        noise_cfg = posterior_config.noise_config
        noise_model = get_noise(noise_cfg.name, **noise_cfg.get_params())

        op_cfg = posterior_config.operator_config
        if isinstance(op_cfg, CompositeOperatorConfig):
            operator = sum(get_operator(op.name, **op.get_params()) for op in op_cfg)
            assert isinstance(operator, CompositeOperator)
        else:
            operator = get_operator(op_cfg.name, **op_cfg.get_params())

        return PosteriorVelocityModelWrapper(
            velocity_predictor=v_model,
            path=self.path,
            operator=operator,
            noise_model=noise_model,
            measurement_y=measurement_y.to(self.device),
            valid_length=posterior_config.valid_length,
            scale=posterior_config.scale,
            num_sampling=posterior_config.num_sampling,
            project=False,  # method=PROJECT already projects internally
            method=PosteriorMethod.PROJECT,
        )

    def _prepare_x_0(
        self,
        shape: tuple[int, int, int],
        x_0: Tensor | None,
        valid_length: Tensor | None,
    ) -> Tensor:
        if x_0 is None:
            x_0 = torch.randn(*shape, device=self.device)
        else:
            x_0 = x_0.to(self.device)
        if valid_length is not None:
            x_0 = FlowModelPL._zero_padding(x_0, valid_length)
        return x_0

    # ------------------------------------------------------------------ #
    # Unified entry point                                                #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        sample_config: SampleConfig,
        condition: dict[str, Tensor],
        cfg_scale: float = 1.0,
        posterior_config: PosteriorSampleConfig | None = None,
        measurement_y: Tensor | None = None,
        x_0: Tensor | None = None,
        **_kwargs,
    ) -> Tensor:
        """Run the wrapped velocity model through ``ODESolver``.

        Args:
            sample_config: batch size, num_step, method, use_ema, data_shape.
            condition: dict of ``[batch_size, 1]`` conditioning tensors.
            cfg_scale: classifier-free-guidance scale (1.0 = disabled).
            posterior_config: if given, wraps with ``PosteriorVelocityModelWrapper``
                (method=PROJECT). Requires ``measurement_y``.
            measurement_y: measurement tensor; shape depends on the operator.
            x_0: optional initial noise ``[B, num_patches, patch_size]``.

        Returns:
            Samples in patched space ``[B, num_patches, patch_size]``.
        """
        if posterior_config is not None and measurement_y is None:
            raise ValueError("posterior_config requires measurement_y to be provided")

        condition = {k: v.to(self.device) for k, v in condition.items()}

        v_model = self._get_base_velocity_model(sample_config.use_ema)
        v_model = self._wrap_conditional(v_model, condition, cfg_scale)
        v_model, valid_length = self._wrap_position_alignment(v_model, condition)
        if posterior_config is not None:
            v_model = self._wrap_posterior(v_model, posterior_config, measurement_y)

        num_patches, patch_size = sample_config.data_shape
        x_0 = self._prepare_x_0(
            shape=(sample_config.batch_size, num_patches, patch_size),
            x_0=x_0,
            valid_length=valid_length if self.create_mask else None,
        )

        solver = ODESolver(velocity_model=v_model)
        with torch.no_grad():
            x_1 = solver.sample(
                x_init=x_0,
                method=sample_config.method.value
                if hasattr(sample_config.method, "value")
                else sample_config.method,
                step_size=1.0 / sample_config.num_step,
            )
        if valid_length is not None and self.create_mask:
            x_1 = FlowModelPL._zero_padding(x_1, valid_length)
        return x_1

    # ------------------------------------------------------------------ #
    # Convenience wrappers                                               #
    # ------------------------------------------------------------------ #

    def _default_sample_config(self, batch_size: int, num_step: int) -> SampleConfig:
        return SampleConfig(
            use_ema=True,
            batch_size=batch_size,
            num_step=num_step,
            data_shape=(self.num_patches, self.patch_size),
        )

    def _expand_condition(
        self, condition: dict[str, Tensor], batch_size: int
    ) -> dict[str, Tensor]:
        """Broadcast a per-profile (batch=1) condition dict to ``batch_size``."""
        out: dict[str, Tensor] = {}
        for k, v in condition.items():
            if v.shape[0] == batch_size:
                out[k] = v.to(self.device)
            elif v.shape[0] == 1:
                out[k] = v.to(self.device).expand(batch_size, -1).contiguous()
            else:
                raise ValueError(
                    f"condition[{k}] has batch dim {v.shape[0]}; "
                    f"expected 1 or {batch_size}"
                )
        return out

    def generate(
        self,
        condition: dict[str, Tensor],
        batch_size: int,
        *,
        cfg_scale: float = 1.0,
        num_step: int = 100,
    ) -> Tensor:
        """Unconditional-from-prior generation in patched space.

        ``condition`` must be broadcastable to ``batch_size``.
        """
        condition = self._expand_condition(condition, batch_size)
        return self.sample(
            sample_config=self._default_sample_config(batch_size, num_step),
            condition=condition,
            cfg_scale=cfg_scale,
        )

    def impute(
        self,
        observed_real: Tensor,
        mask_real: Tensor,
        condition: dict[str, Tensor],
        *,
        num_samples: int = 1,
        num_step: int = 100,
        cfg_scale: float = 1.0,
    ) -> Tensor:
        """Impute missing values given a single real-space profile.

        Args:
            observed_real: ``[seq_length]`` or ``[1, seq_length]`` real-space
                profile with missing entries zeroed (or any value; the mask
                decides what is observed).
            mask_real: ``[seq_length]`` or ``[1, seq_length]`` float/bool mask
                (1 = observed, 0 = missing).
            condition: condition dict for a single profile (batch 1).
            num_samples: number of independent posterior samples to draw.
            num_step: ODE steps.
            cfg_scale: classifier-free-guidance scale.

        Returns:
            Imputed samples in real space ``[num_samples, seq_length]``.
        """
        observed_real = observed_real.view(-1).to(self.device)
        mask_real = mask_real.view(-1).to(self.device)
        if observed_real.shape[0] != self.seq_length:
            raise ValueError(
                f"observed_real has {observed_real.shape[0]} timesteps; "
                f"expected {self.seq_length}"
            )
        if mask_real.shape[0] != self.seq_length:
            raise ValueError(
                f"mask_real has {mask_real.shape[0]} timesteps; "
                f"expected {self.seq_length}"
            )

        mask_batch = mask_real.unsqueeze(0).expand(num_samples, -1).bool()
        observed_batch = (
            (observed_real * mask_real).unsqueeze(0).expand(num_samples, -1)
        )

        posterior_config = PosteriorSampleConfig(
            noise_config={"name": "gaussian", "sigma": 0.01},
            operator_config=OperatorConfig(
                name="inpainting", mask=mask_batch.cpu().tolist()
            ),
            method="project",
        )

        expanded_condition = self._expand_condition(condition, num_samples)
        x_1 = self.sample(
            sample_config=self._default_sample_config(num_samples, num_step),
            condition=expanded_condition,
            cfg_scale=cfg_scale,
            posterior_config=posterior_config,
            measurement_y=observed_batch,
        )
        return rearrange(x_1, "b l c -> b (l c)")

    def super_resolve(
        self,
        low_res_real: Tensor,
        scale_factor: int,
        condition: dict[str, Tensor],
        *,
        num_samples: int = 1,
        num_step: int = 100,
        cfg_scale: float = 1.0,
    ) -> Tensor:
        """Upsample a low-resolution profile to the model's native resolution.

        Args:
            low_res_real: ``[low_res_len]`` or ``[1, low_res_len]`` where
                ``low_res_len * scale_factor == seq_length``.
            scale_factor: integer upsampling factor.
            condition: condition dict for a single profile (batch 1).
            num_samples: number of stochastic SR samples.
            num_step: ODE steps.

        Returns:
            Super-resolved samples in real space ``[num_samples, seq_length]``.
        """
        low_res_real = low_res_real.view(-1).to(self.device)
        low_res_len = low_res_real.shape[0]
        if low_res_len * scale_factor != self.seq_length:
            raise ValueError(
                f"low_res_len ({low_res_len}) * scale_factor ({scale_factor}) "
                f"!= seq_length ({self.seq_length})"
            )

        measurement_y = low_res_real.unsqueeze(0).expand(num_samples, -1)

        posterior_config = PosteriorSampleConfig(
            noise_config={"name": "gaussian", "sigma": 0.01},
            operator_config=OperatorConfig(
                name="super_resolution", scale_factor=scale_factor
            ),
            method="project",
        )

        expanded_condition = self._expand_condition(condition, num_samples)
        x_1 = self.sample(
            sample_config=self._default_sample_config(num_samples, num_step),
            condition=expanded_condition,
            cfg_scale=cfg_scale,
            posterior_config=posterior_config,
            measurement_y=measurement_y,
        )
        return rearrange(x_1, "b l c -> b (l c)")
