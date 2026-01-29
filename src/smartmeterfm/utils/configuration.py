from dataclasses import dataclass, field

from easy_ml_config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    dataset: str
    root: str
    resolution: str
    load: bool
    normalize: bool
    normalize_method: str
    pit: bool
    shuffle: bool
    vectorize: bool
    style_vectorize: str
    vectorize_window_size: int
    train_season: str
    val_season: str
    target_labels: str
    scaling_factor: list[float] | None = None


@dataclass
class CossmicDataConfig(DataConfig):
    subdataset_names: str = "grid_import_residential"
    val_area: str = "all"


@dataclass
class ModelConfig(BaseConfig):
    model_class: str
    dim_base: int
    conditioning: bool = False
    cond_dropout: float = 0.1
    dropout: float = 0.1
    num_attn_head: int = 4
    dim_feedforward: int = 2048
    learn_variance: bool = False
    num_in_channel: int = -1  # -1: uninitialized
    seq_length: int = -1  # -1: uninitialized

    load_time_id: str | None = None
    load_milestone: int | None = None
    resume: bool = False
    freeze_layers: bool = False


@dataclass
class TransformerConfig(ModelConfig):
    num_encoder_layer: int = 6
    num_decoder_layer: int = 6


@dataclass
class UNetConfig(ModelConfig):
    dim_mult: tuple[int, ...] | list[int] = (1, 2, 4, 8)


@dataclass
class FMConfig(BaseConfig):
    "Flow Matching Configuration"

    prediction_type: str

    def __post_init__(self):
        self.prediction_type = self.prediction_type.lower()
        assert self.prediction_type in [
            "velocity",
            "x0",
            "x1",
        ]


@dataclass
class SampleConfig(BaseConfig):
    num_sample: int = 512
    val_batch_size: int = 256
    num_sampling_step: int = 50
    method: str = "euler"
    cfg_scale: float = 1.0

    def __post_init__(self):
        self.method = self.method.lower()


@dataclass
class TrainConfig(BaseConfig):
    batch_size: int
    val_sample_config: SampleConfig = field(default_factory=SampleConfig)
    lr: float = 1e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    gradient_accumulate_every: int = 1
    ema_update_every: int = 5
    ema_decay: float = 0.9999
    amp: bool = True
    mixed_precision_type: str = "fp16"
    split_batches: bool = True
    num_train_step: int = 50000
    save_and_sample_every: int = 50000
    val_every: int = 1250
    val_batch_size: int = 256


@dataclass
class ExperimentConfig(BaseConfig):
    exp_id: str
    data: DataConfig
    model: TransformerConfig
    flow: FMConfig
    train: TrainConfig
    sample: SampleConfig
    log_wandb: bool = False
    log_mlflow: bool = False
    time_id: str | None = None
    wandb_id: str | None = None  # None: no wandb logging or has not been initialized
    wandb_project: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_run_id: str | None = None


@dataclass
class IntegerEmbedderArgs(BaseConfig):
    "reference: models.nn_components.IntegerEmbedder"

    num_embedding: int
    dim_embedding: int
    dropout: float = 0.1
    quantize: bool = False
    quantize_max_val: float = 20000.0
    quantize_min_val: float = 0.0


@dataclass
class PositionEmbedderArgs(BaseConfig):
    "reference: models.embedders.PositionEmbedder"

    dim_embedding: int


if __name__ == "__main__":
    model_config = ModelConfig.from_yaml("model_config.yaml")
    train_config = TrainConfig(batch_size=32)
    sample_config = SampleConfig(num_sample=100)

    # exp_config = ExperimentConfig(
    #     exp_id=1, model=model_config, data=train=train_config, sample=sample_config
    # )
    # print(exp_config.model.num_layer)
    # exp_config.to_yaml("exp_config.yaml")
    pass


def save_config(exp_config: ExperimentConfig, time_id: str) -> None:
    "save config to yaml"
    import os

    os.makedirs("results/configs", exist_ok=True)
    exp_config.to_yaml(f"results/configs/exp_config_{time_id}.yaml")
