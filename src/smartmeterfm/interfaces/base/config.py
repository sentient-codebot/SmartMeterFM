from enum import Enum
from typing import Annotated

import torch
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


type TorchDevice = torch.device | str


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_path: str
    devices: TorchDevice | list[TorchDevice] = "cpu"
    s3_bucket_name: str | None = None

    @model_validator(mode="after")
    def validate_bucket_name(self) -> "ModelConfig":
        if self.model_path.startswith("s3://") and self.s3_bucket_name is None:
            raise ValueError(
                "s3_bucket_name must be provided when \
                model_path starts with 's3://'"
            )
        return self

    @model_validator(mode="after")
    def validate_device(self) -> "ModelConfig":
        if isinstance(self.devices, str):
            self.devices = [torch.device(self.devices)]
        elif isinstance(self.devices, torch.device):
            self.devices = [self.devices]
        elif isinstance(self.devices, list):
            self.devices = [torch.device(d) for d in self.devices]

        # if more than one device, check if they are the same type (must be cuda)
        if isinstance(self.devices, list):
            device_type = self.devices[0].type
            for d in self.devices:
                if d.type != device_type:
                    raise ValueError(
                        "All devices must be of the same type \
                        (e.g., all 'cuda'')"
                    )
        return self

    @field_serializer("devices")
    def serialize_devices(self, v: list[torch.device]) -> list[str]:
        return [str(d) for d in v]


class SolverMethod(str, Enum):
    EULER = "euler"
    MIDPOINT = "midpoint"


class SDEConfig(BaseModel):
    ode_threshold: Annotated[float, Field(ge=0.0, le=1.0)]


def default_sde_config_factory() -> SDEConfig:
    return SDEConfig(ode_threshold=1.0)


class SampleConfig(BaseModel):
    use_ema: bool = True
    batch_size: int = 128
    num_step: int = 1000
    method: SolverMethod = SolverMethod.EULER
    use_sde: bool = False
    data_shape: tuple
    sde_config: SDEConfig | None = None

    @model_validator(mode="after")
    def validate_sde_config(self) -> "SampleConfig":
        if self.use_sde and self.sde_config is None:
            self.sde_config = default_sde_config_factory()
        elif not self.use_sde:
            self.sde_config = None
        return self
