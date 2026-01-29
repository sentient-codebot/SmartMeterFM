from jaxtyping import Float
from torch import Tensor

from ..nn_components import (
    CombinedEmbedder,
    ContextEmbedder,
    IntegerEmbedder,
    PositionEmbedder,
)
from ._registry import logger, register_embedder


try:
    from large_customer_data_utils.advanced_transforms import (  # type: ignore
        AddMaxValueLabel,
        AddMeanValueLabel,
        AddMinValueLabel,
        AttributeCategoryEncoder,
        AttributeToTensor,
        CombineAttributes,
        Compose,
        FeatureToTensor,
        GenerationTypeCombinedCategoryEncoder,
        IntegerOffset,
        NormalizeAttributes,
        ProfileCategoryEncoder,
        YearMonthSpliter,
    )

    HAS_ALLIANDER = True
except ImportError:
    HAS_ALLIANDER = False

missing_alliander_error = ImportError(
    "alliander large customer data utils not installed. "
    "Please install it to use this embedder."
)


# label embedder is always "relaxed"
@register_embedder(name="alliander_label")
class AllianderLargeCustomerLabelEmbedder(CombinedEmbedder):
    """High-level wrapper for create the entire embedder for alliander large\
        customer data.
    -> CombinedEmbedder

    Args:
        - year: dim_embedding (int)
        - month:
            - dim_embedding (int)
            - num_embedding
        - first_day_of_week:
            - num_embedding
            - dim_embedding
            - drop_out [optional]
            - quantize [optional]
            - ...
        - month_length:
            - num_embedding
            - dim_embedding
        - profielcategorie:
            - num_embedding
            - dim_embedding
        - baseload_profile
            - dim_embedding
            - num_embedding
        - generation_type_category
            - dim_embedding
            - num_embedding

    Input:
        labels: Int[Tensor, "batch 5"]

    Output:
        embedding: Float[Tensor, "batch dim_embedding"]
    """

    def __init__(self, **kwargs):
        if not HAS_ALLIANDER:
            raise missing_alliander_error

        year_embedder = PositionEmbedder(**kwargs["year"])
        month_embedder = IntegerEmbedder(**kwargs["month"])
        first_day_of_week_embedder = IntegerEmbedder(**kwargs["first_day_of_week"])
        month_length_embedder = IntegerEmbedder(**kwargs["month_length"])
        baseload_profile_embedder = IntegerEmbedder(**kwargs["baseload_profile"])
        generation_type_category = IntegerEmbedder(**kwargs["generation_type_category"])
        # profielcategorie_embedder = IntegerEmbedder(**kwargs["profielcategorie"])

        super().__init__(
            dict_embedder={
                "year": year_embedder,
                "month": month_embedder,
                "first_day_of_week": first_day_of_week_embedder,
                "month_length": month_length_embedder,
                "BASELOAD_PROFILE": baseload_profile_embedder,
                "GENERATION_TYPE_CATEGORY": generation_type_category,
            }
        )

    def forward(
        self, dict_labels: dict[str, Float[Tensor, "batch dim_label"]], **kwargs
    ):
        return super().forward(dict_labels, **kwargs)


register_embedder(name="alliander_label_relaxed")(AllianderLargeCustomerLabelEmbedder)


# used for restricted generation for vae/gan. not used by fm.
@register_embedder(name="alliander_label_rich")
class AllianderLargeCustomerRichLabelEmbedder(CombinedEmbedder):
    """High-level wrapper for create the entire embedder for alliander large\
        customer data.
    -> CombinedEmbedder

    Args:
        - year: dim_embedding (int)
        - month:
            - dim_embedding (int)
            - num_embedding
        - first_day_of_week:
            - num_embedding
            - dim_embedding
            - drop_out [optional]
            - quantize [optional]
            - ...
        - month_length:
            - num_embedding
            - dim_embedding
        - profielcategorie:
            - num_embedding
            - dim_embedding
        - baseload_profile
            - dim_embedding
            - num_embedding
        - generation_type_category
            - dim_embedding
            - num_embedding

    Input:
        labels: Int[Tensor, "batch 5"]

    Output:
        embedding: Float[Tensor, "batch dim_embedding"]
    """

    def __init__(self, **kwargs):
        if not HAS_ALLIANDER:
            raise missing_alliander_error

        year_embedder = PositionEmbedder(**kwargs["year"])
        month_embedder = IntegerEmbedder(**kwargs["month"])
        first_day_of_week_embedder = IntegerEmbedder(**kwargs["first_day_of_week"])
        month_length_embedder = IntegerEmbedder(**kwargs["month_length"])
        baseload_profile_embedder = IntegerEmbedder(**kwargs["baseload_profile"])
        generation_type_category = IntegerEmbedder(**kwargs["generation_type_category"])
        # profielcategorie_embedder = IntegerEmbedder(**kwargs["profielcategorie"])
        max_value_embedder = PositionEmbedder(**kwargs["max_value"])
        min_value_embedder = PositionEmbedder(**kwargs["min_value"])
        mean_value_embedder = PositionEmbedder(**kwargs["mean_value"])

        super().__init__(
            dict_embedder={
                "year": year_embedder,
                "month": month_embedder,
                "first_day_of_week": first_day_of_week_embedder,
                "month_length": month_length_embedder,
                "BASELOAD_PROFILE": baseload_profile_embedder,
                "GENERATION_TYPE_CATEGORY": generation_type_category,
                "max_value": max_value_embedder,
                "min_value": min_value_embedder,
                "mean_value": mean_value_embedder,
            }
        )

    def forward(
        self, dict_labels: dict[str, Float[Tensor, "batch dim_label"]], **kwargs
    ):
        return super().forward(dict_labels, **kwargs)


@register_embedder(name="alliander_context_relaxed")
class AllianderLargeCustomerRelaxedContextEmbedder(ContextEmbedder):
    """High-level wrapper for create the entire embedder for alliander large\
        customer data. A relaxed version without max, mean, min values.
    -> ContextEmbedder

    Args:
        - year: dim_embedding (int)
        - month:
            - dim_embedding (int)
            - num_embedding
        - first_day_of_week:
            - num_embedding
            - dim_embedding
            - drop_out [optional]
            - quantize [optional]
            - ...
        - month_length:
            - num_embedding
            - dim_embedding
        - profielcategorie:
            - num_embedding
            - dim_embedding
        - baseload_profile
            - dim_embedding
            - num_embedding
        - generation_type_category
            - dim_embedding
            - num_embedding


    Input:
        labels: dict[str, Tensor]

    Output:
        embedding: Float[Tensor, "batch seq dim_embedding"]
    """

    def __init__(self, **kwargs):
        if not HAS_ALLIANDER:
            raise missing_alliander_error

        year_embedder = PositionEmbedder(**kwargs["year"])
        month_embedder = IntegerEmbedder(**kwargs["month"])
        first_day_of_week_embedder = IntegerEmbedder(**kwargs["first_day_of_week"])
        month_length_embedder = IntegerEmbedder(**kwargs["month_length"])
        baseload_profile_embedder = IntegerEmbedder(**kwargs["baseload_profile"])
        generation_type_category = IntegerEmbedder(**kwargs["generation_type_category"])

        super().__init__(
            dict_embedder={
                "year": year_embedder,
                "month": month_embedder,
                "first_day_of_week": first_day_of_week_embedder,
                "month_length": month_length_embedder,
                "BASELOAD_PROFILE": baseload_profile_embedder,
                "GENERATION_TYPE_CATEGORY": generation_type_category,
            }
        )

    def forward(
        self, dict_labels: dict[str, Float[Tensor, "batch dim_label"]], **kwargs
    ):
        return super().forward(dict_labels, **kwargs)


@register_embedder(name="alliander_context")
class AllianderLargeCustomerContextEmbedder(ContextEmbedder):
    """High-level wrapper for create the entire embedder for alliander large\
        customer data.
    -> ContextEmbedder

    Args:
        - year: dim_embedding (int)
        - month:
            - dim_embedding (int)
            - num_embedding
        - first_day_of_week:
            - num_embedding
            - dim_embedding
            - drop_out [optional]
            - quantize [optional]
            - ...
        - month_length:
            - num_embedding
            - dim_embedding
        - profielcategorie:
            - num_embedding
            - dim_embedding
        - baseload_profile
            - dim_embedding
            - num_embedding
        - generation_type_category
            - dim_embedding
            - num_embedding

        - max_value
        - min_value
        - mean_value

        [del] - profielcategori:
            - num_embedding
            - dim_embedding

    Input:
        labels: dict[str, Tensor]

    Output:
        embedding: Float[Tensor, "batch seq dim_embedding"]
    """

    def __init__(self, **kwargs):
        if not HAS_ALLIANDER:
            raise missing_alliander_error

        year_embedder = PositionEmbedder(**kwargs["year"])
        month_embedder = IntegerEmbedder(**kwargs["month"])
        first_day_of_week_embedder = IntegerEmbedder(**kwargs["first_day_of_week"])
        month_length_embedder = IntegerEmbedder(**kwargs["month_length"])
        baseload_profile_embedder = IntegerEmbedder(**kwargs["baseload_profile"])
        generation_type_category = IntegerEmbedder(**kwargs["generation_type_category"])
        # profielcategorie_embedder = IntegerEmbedder(**kwargs["profielcategorie"])
        max_value_embedder = PositionEmbedder(**kwargs["max_value"])
        min_value_embedder = PositionEmbedder(**kwargs["min_value"])
        mean_value_embedder = PositionEmbedder(**kwargs["mean_value"])

        super().__init__(
            dict_embedder={
                "year": year_embedder,
                "month": month_embedder,
                "first_day_of_week": first_day_of_week_embedder,
                "month_length": month_length_embedder,
                "BASELOAD_PROFILE": baseload_profile_embedder,
                "GENERATION_TYPE_CATEGORY": generation_type_category,
                "max_value": max_value_embedder,
                "min_value": min_value_embedder,
                "mean_value": mean_value_embedder,
            }
        )

    def forward(
        self, dict_labels: dict[str, Float[Tensor, "batch dim_label"]], **kwargs
    ):
        return super().forward(dict_labels, **kwargs)


def create_temporal_transform_pipeline():
    """
    Create a transform pipeline for temporal features.
    """
    if not HAS_ALLIANDER:
        raise missing_alliander_error

    # temporal features
    temporal_transform = Compose(
        [
            YearMonthSpliter("year_month"),
            IntegerOffset("year", 1950),  # already tensor
            FeatureToTensor("year"),  # also adjusts the shape
            IntegerOffset("month", 1),  # 1-12 -> 0-11
            FeatureToTensor("month"),
            IntegerOffset("month_length", 28),  # already tensor
            FeatureToTensor("month_length"),  # also adjusts the shape
            FeatureToTensor("first_day_of_week"),  # range [0,6]
            # Encode day of week as cyclical feature
            # CyclicalTimeEncoder('first_day_of_week', period=7),
            # OneHotFirstDayOfWeek("first_day_of_week")
            # Normalize month length (not used)
            # MonthLengthNormalizer('month_length', min_days=28, max_days=31)
        ]
    )
    # additional labels
    additional_transform = Compose(
        [
            AddMaxValueLabel("max_value"),
            AddMinValueLabel("min_value"),
            AddMeanValueLabel("mean_value"),
        ]
    )
    return Compose([temporal_transform, additional_transform])


def create_attribute_transform_pipeline(
    attr_keys: str | list[str] | None,
    attr_stats: dict | None,
    normalize: bool = False,
    combine=False,
):
    """
    Create a transform pipeline for attributes.

    - ProfileCategoryEncoder
    - [optional] Combine

    Args:
        attr_keys: attr_keys to combine if needed.
        attr_stats: Statistics for each attribute from get_attribute_stats()
        combine: Whether to combine attributes into a single vector

    Returns:
        AttributeTransform pipeline
    """
    if not HAS_ALLIANDER:
        raise missing_alliander_error

    # Create normalizers for each attribute
    transforms = []
    attr_keys = attr_keys or list(attr_stats.keys())

    # Add normalizers
    if normalize:
        try:
            attr_mins = {key: attr_stats[key]["min"] for key in attr_keys}
            attr_maxs = {key: attr_stats[key]["max"] for key in attr_keys}
            transforms.append(NormalizeAttributes(attr_keys, attr_mins, attr_maxs))
        except KeyError as err:
            logger.warning(f"failed to add AttributeNormalizer {err}")

    # Add profilecategorie mapper
    transforms.extend(
        [
            ProfileCategoryEncoder(attr_key="PROFIELCATEGORIE"),
            AttributeToTensor(attr_keys="PROFIELCATEGORIE"),
            AttributeCategoryEncoder(attr_key="BASELOAD_PROFILE"),
            AttributeToTensor(attr_keys="BASELOAD_PROFILE"),
            GenerationTypeCombinedCategoryEncoder(
                ["INSTALLATIE_SOORT", "INSTALLATIE_SOORT_TL"],
                "GENERATION_TYPE_CATEGORY",
            ),
        ]
    )

    # Combine attributes if requested
    if combine and attr_keys:
        transforms.append(CombineAttributes(attr_keys, "combined_attributes"))

    return Compose(transforms)
