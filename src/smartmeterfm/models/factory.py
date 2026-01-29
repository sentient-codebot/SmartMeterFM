from typing import Any

from .nn_components import ConcatEmbedder, IntegerEmbedder


def make_hp_month_embedder(dim_base: int) -> IntegerEmbedder:
    return IntegerEmbedder(12, dim_base, 0.1, False)


# deprecated
def make_concat_embedder(dict_args: dict[str, dict[str, Any]]) -> ConcatEmbedder:
    all_embedders = []
    for label_name, label_args in dict_args.items():
        if label_name == "month":
            embedder = make_hp_month_embedder(label_args["dim_base"])
        else:
            raise ValueError(f"Unknown label name: {label_name}")
        all_embedders.append(embedder)
    concat_embedder = ConcatEmbedder(all_embedders)
    return concat_embedder
