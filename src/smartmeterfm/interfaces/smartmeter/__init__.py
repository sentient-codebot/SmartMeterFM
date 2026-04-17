"""Concrete ``ModelInterface`` for SmartMeterFM Flow Matching checkpoints.

Public surface:

    from smartmeterfm.interfaces.smartmeter import SmartMeterFMModel
    from smartmeterfm.interfaces.base import (
        ModelConfig,
        SampleConfig,
    )
    from smartmeterfm.interfaces.base.posterior import (
        NoiseConfig,
        OperatorConfig,
        PosteriorSampleConfig,
    )
"""

from .model import SmartMeterFMModel


__all__ = ["SmartMeterFMModel"]
