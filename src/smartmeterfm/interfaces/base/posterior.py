from collections.abc import Sequence
from enum import Enum
from typing import Literal

from pydantic import (
    BaseModel,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


# Noise
class NoiseConfig(BaseModel):
    name: Literal["gaussian", "laplace"]
    sigma: PositiveFloat | Sequence[PositiveFloat] | None = None  # gaussian
    b: PositiveFloat | Sequence[PositiveFloat] | None = None  # laplace

    @model_validator(mode="after")
    def check_noise_params(self):
        if self.name == "gaussian" and self.sigma is None:
            raise ValueError("sigma must be set for gaussian noise")
        if self.name == "laplace" and self.b is None:
            raise ValueError("b must be set for laplace noise")
        return self

    def get_params(self) -> dict:
        if self.name == "gaussian":
            return {"sigma": self.sigma}
        elif self.name == "laplace":
            return {"b": self.b}
        else:
            raise ValueError(f"Unknown noise type: {self.name}")


# Operator
class OperatorType(str, Enum):
    LINEAR = "linear"
    DENOISE = "denoise"
    SUPER_RESOLUTION = "super_resolution"
    AVERAGE = "average"
    INPAINTING = "inpainting"
    PARTIAL_SUM = "partial_sum"
    MAX = "max"
    SMOOTH_MAX = "smooth_max"
    SMOOTH_MIN = "smooth_min"
    LSE_MAX = "lse_max"
    LSE_MIN = "lse_min"
    SOFT_QUANTILE = "soft_quantile"
    LDN_ODN = "ldn_odn"


class OperatorConfig(BaseModel):
    name: OperatorType
    sequence_length: int | None = None
    mask: Sequence[int] | Sequence[Sequence[int]] | None = None
    tau: PositiveFloat | None = None
    q: PositiveFloat | None = None
    scale_factor: PositiveInt | None = None  # for super resolution

    @model_validator(mode="after")
    def check_operator_params(self):
        if self.name == "average" and self.sequence_length is None:
            raise ValueError("sequence_length must be set for average operator")
        if self.name == "inpainting" and self.mask is None:
            raise ValueError("mask must be set for inpainting operator")
        if self.name == "smooth_max" and self.tau is None:
            pass  # use default tau
        if self.name == "smooth_min" and self.tau is None:
            pass  # use default tau
        if self.name == "soft_quantile" and self.q is None:
            raise ValueError("q must be set for soft_quantile operator")
        if self.name == "super_resolution" and self.scale_factor is None:
            raise ValueError("scale_factor must be set for super resolution operator")
        return self

    def get_params(self) -> dict:
        if self.name == "average":
            return {"sequence_length": self.sequence_length}
        elif self.name == "inpainting":
            return {"mask": self.mask}
        elif self.name == "smooth_max" and self.tau is not None:
            return {"tau": self.tau}
        elif self.name == "smooth_min" and self.tau is not None:
            return {"tau": self.tau}
        elif self.name == "soft_quantile" and self.q is not None:
            return {"q": self.q}
        elif self.name == "ldn_odn":
            return {"sequence_length": self.sequence_length}
        elif self.name == "super_resolution" and self.scale_factor is not None:
            return {"scale_factor": self.scale_factor}
        else:
            return {}


class CompositeOperatorConfig(BaseModel):
    operators: Sequence[OperatorConfig]
    dim: int = 1  # Dimension for concatenation

    def get_params(self) -> list[tuple[str, dict]]:
        params = []
        for _op in self.operators:
            params.append((_op.name, _op.get_params()))
        return params

    def __getitem__(self, index: int) -> OperatorConfig:
        return self.operators[index]

    def __len__(self) -> int:
        return len(self.operators)

    def __contains__(self, item: OperatorConfig) -> bool:
        return item in self.operators

    def __iter__(self):
        return iter(self.operators)


class PosteriorSampleConfig(BaseModel):
    scale: float = 1.0
    num_sampling: int = 1
    project: bool = False
    noise_config: NoiseConfig
    operator_config: OperatorConfig | CompositeOperatorConfig | Sequence[OperatorConfig]
    valid_length: PositiveInt | None = None
    method: Literal["dps", "project", "gd"] = "dps"

    @field_validator("operator_config", mode="before")
    @classmethod
    def validate_operator_config(
        cls: type,
        v: OperatorConfig | CompositeOperatorConfig | Sequence[OperatorConfig],
    ) -> OperatorConfig | CompositeOperatorConfig:
        if isinstance(v, OperatorConfig):
            return v
        elif isinstance(v, CompositeOperatorConfig):
            return v
        elif isinstance(v, list | tuple):
            return CompositeOperatorConfig(operators=v)
        else:
            raise ValueError(
                "operator_config must be an instance of OperatorConfig or CompositeOperatorConfig"
            )


def main():
    noise_config = NoiseConfig(name="gaussian", sigma=0.1)
    print(noise_config)


if __name__ == "__main__":
    main()
