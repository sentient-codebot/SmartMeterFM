import argparse

from smartmeterfm.utils.configuration import ExperimentConfig


def inference_parser() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        prog="Test - Flow Matching Energy Data",
        description="Run test metrics for generated data \
            from trianed flow matcbing model",
    )
    parser.add_argument(
        "--load_time_id",
        type=str,
        help="Time id of the experiment to load",
        required=True,
    )
    args, _ = parser.parse_known_args()
    try:
        config = ExperimentConfig.from_yaml(
            f"results/configs/exp_config_{args.load_time_id}.yaml"
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Config file not found for time id: {args.load_time_id}"
        ) from e

    config.time_id = args.load_time_id

    return config
