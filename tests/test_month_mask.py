"""Tests for start_pos / month_length mask creation logic.

Covers:
- get_start_pos: convert first_day_of_week -> timestep offset
- _convert_offset_month_length: month_length (0-3) -> valid timestep count
- _create_loss_mask: binary mask (1=valid, 0=padding) for variable-length months
- Calendar derivation: year+month -> first_day_of_week, month_length
- End-to-end mask pipeline integration
"""

import calendar

import pytest
import torch

from smartmeterfm.models.baselines.cond_gen.gan import GANModelPL
from smartmeterfm.models.baselines.cond_gen.vae import VAEModelPL
from smartmeterfm.models.flow import FlowModelPL
from smartmeterfm.models.nn_components import get_start_pos


# ---------------------------------------------------------------------------
# TestGetStartPos
# ---------------------------------------------------------------------------


class TestGetStartPos:
    @pytest.mark.parametrize("day", range(7))
    @pytest.mark.parametrize("steps_per_day", [24, 96])
    def test_int_input(self, day, steps_per_day):
        result = get_start_pos(day, steps_per_day)
        assert result == day * steps_per_day

    def test_tensor_batch(self):
        days = torch.tensor([0, 3, 6])
        result = get_start_pos(days, 96)
        expected = torch.tensor([0, 288, 576])
        assert torch.equal(result, expected)

    def test_tensor_scalar(self):
        result = get_start_pos(torch.tensor(3), 96)
        assert result == 288

    def test_boundary_zero(self):
        assert get_start_pos(0, 96) == 0

    def test_boundary_six(self):
        assert get_start_pos(6, 96) == 576

    def test_invalid_negative_int(self):
        with pytest.raises(ValueError, match="first_day_of_week"):
            get_start_pos(-1, 96)

    def test_invalid_seven_int(self):
        with pytest.raises(ValueError, match="first_day_of_week"):
            get_start_pos(7, 96)

    def test_invalid_tensor_negative(self):
        with pytest.raises(ValueError, match="first_day_of_week"):
            get_start_pos(torch.tensor([-1]), 96)

    def test_invalid_tensor_seven(self):
        with pytest.raises(ValueError, match="first_day_of_week"):
            get_start_pos(torch.tensor([7]), 96)

    def test_invalid_tensor_mixed(self):
        with pytest.raises(ValueError, match="first_day_of_week"):
            get_start_pos(torch.tensor([0, 7]), 96)

    def test_invalid_steps_zero(self):
        with pytest.raises(ValueError, match="steps_per_day"):
            get_start_pos(0, 0)

    def test_invalid_steps_negative(self):
        with pytest.raises(ValueError, match="steps_per_day"):
            get_start_pos(0, -1)


# ---------------------------------------------------------------------------
# TestConvertOffsetMonthLength
# ---------------------------------------------------------------------------

ALL_MODEL_CLASSES = [FlowModelPL, VAEModelPL, GANModelPL]


class TestConvertOffsetMonthLength:
    @pytest.mark.parametrize(
        "month_length, expected_days",
        [(0, 28), (1, 29), (2, 30), (3, 31)],
    )
    def test_scalar_96_steps(self, month_length, expected_days):
        result = FlowModelPL._convert_offset_month_length(month_length, 28, 96)
        assert result == expected_days * 96

    @pytest.mark.parametrize(
        "month_length, expected_days",
        [(0, 28), (1, 29), (2, 30), (3, 31)],
    )
    def test_scalar_24_steps(self, month_length, expected_days):
        result = FlowModelPL._convert_offset_month_length(month_length, 28, 24)
        assert result == expected_days * 24

    def test_tensor_batch(self):
        ml = torch.tensor([0, 1, 2, 3])
        result = FlowModelPL._convert_offset_month_length(ml, 28, 96)
        expected = torch.tensor([2688, 2784, 2880, 2976])
        assert torch.equal(result, expected)

    def test_tensor_shaped_like_training(self):
        """Tensor with shape [batch, 1] as used in training before .squeeze(1)."""
        ml = torch.tensor([[0], [3]])
        result = FlowModelPL._convert_offset_month_length(ml, 28, 96)
        expected = torch.tensor([[2688], [2976]])
        assert torch.equal(result, expected)

    @pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
    def test_all_classes_identical(self, cls):
        ml = torch.tensor([0, 1, 2, 3])
        result = cls._convert_offset_month_length(ml, 28, 96)
        expected = torch.tensor([2688, 2784, 2880, 2976])
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# TestCreateLossMask
# ---------------------------------------------------------------------------


class TestCreateLossMask:
    def test_shape(self):
        mask = FlowModelPL._create_loss_mask(torch.tensor([2688]), 2976)
        assert mask.shape == (1, 2976)

    def test_31_day_all_ones(self):
        mask = FlowModelPL._create_loss_mask(torch.tensor([2976]), 2976)
        assert torch.all(mask == 1.0)

    def test_28_day_trailing_zeros(self):
        mask = FlowModelPL._create_loss_mask(torch.tensor([2688]), 2976)
        assert torch.all(mask[0, :2688] == 1.0)
        assert torch.all(mask[0, 2688:] == 0.0)

    def test_mixed_batch(self):
        valid = torch.tensor([2688, 2976])
        mask = FlowModelPL._create_loss_mask(valid, 2976)
        # 28-day month
        assert torch.all(mask[0, :2688] == 1.0)
        assert torch.all(mask[0, 2688:] == 0.0)
        # 31-day month
        assert torch.all(mask[1] == 1.0)

    def test_all_four_lengths(self):
        valid = torch.tensor([2688, 2784, 2880, 2976])
        mask = FlowModelPL._create_loss_mask(valid, 2976)
        assert mask.shape == (4, 2976)
        for i in range(4):
            assert torch.all(mask[i, : valid[i]] == 1.0)
            if valid[i] < 2976:
                assert torch.all(mask[i, valid[i] :] == 0.0)

    def test_sum_equals_valid_length(self):
        valid = torch.tensor([2688, 2784, 2880, 2976])
        mask = FlowModelPL._create_loss_mask(valid, 2976)
        assert torch.equal(mask.sum(dim=1).long(), valid)

    def test_values_binary(self):
        valid = torch.tensor([2688, 2976])
        mask = FlowModelPL._create_loss_mask(valid, 2976)
        assert torch.all((mask == 0.0) | (mask == 1.0))

    def test_dtype_float32(self):
        mask = FlowModelPL._create_loss_mask(torch.tensor([2688]), 2976)
        assert mask.dtype == torch.float32

    def test_24_steps(self):
        # 28 days * 24 steps = 672, full = 31 * 24 = 744
        mask = FlowModelPL._create_loss_mask(torch.tensor([672, 744]), 744)
        assert torch.all(mask[0, :672] == 1.0)
        assert torch.all(mask[0, 672:] == 0.0)
        assert torch.all(mask[1] == 1.0)

    @pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
    def test_all_classes_identical(self, cls):
        valid = torch.tensor([2688, 2784, 2880, 2976])
        mask = cls._create_loss_mask(valid, 2976)
        ref = FlowModelPL._create_loss_mask(valid, 2976)
        assert torch.equal(mask, ref)


# ---------------------------------------------------------------------------
# TestCollateDerivation
# ---------------------------------------------------------------------------


class TestCollateDerivation:
    """Test the calendar.monthrange derivation logic used in collate functions.

    The collate function (train_flow.py:73-86) derives:
        weekday, days = calendar.monthrange(year, month_1indexed)
        first_day_of_week = weekday      # 0=Monday, 6=Sunday
        month_length = days - 28         # 0-3
    """

    @staticmethod
    def _derive(year: int, month_0indexed: int):
        weekday, days = calendar.monthrange(year, month_0indexed + 1)
        return weekday, days - 28

    def test_feb_2023_non_leap(self):
        fdow, ml = self._derive(2023, 1)
        assert fdow == 2  # Wednesday
        assert ml == 0  # 28 days

    def test_feb_2024_leap(self):
        fdow, ml = self._derive(2024, 1)
        assert fdow == 3  # Thursday
        assert ml == 1  # 29 days

    def test_jan_2024(self):
        fdow, ml = self._derive(2024, 0)
        assert fdow == 0  # Monday
        assert ml == 3  # 31 days

    def test_apr_2024(self):
        fdow, ml = self._derive(2024, 3)
        assert fdow == 0  # Monday
        assert ml == 2  # 30 days

    @pytest.mark.parametrize("year", [2018, 2019, 2020])
    def test_wpuq_years_ranges(self, year):
        """All months in WPuQ data years produce valid ranges."""
        for month_0 in range(12):
            fdow, ml = self._derive(year, month_0)
            assert 0 <= fdow <= 6, f"fdow={fdow} for {year}-{month_0 + 1}"
            assert 0 <= ml <= 3, f"month_length={ml} for {year}-{month_0 + 1}"


# ---------------------------------------------------------------------------
# TestMaskPipelineIntegration
# ---------------------------------------------------------------------------


class TestMaskPipelineIntegration:
    """End-to-end: calendar -> month_length -> valid_length -> mask -> zeroed padding."""

    STEPS_PER_DAY = 96
    FULL_LENGTH = 31 * 96  # 2976

    def _pipeline(self, year: int, month_0: int):
        weekday, days = calendar.monthrange(year, month_0 + 1)
        month_length = days - 28
        start_pos = get_start_pos(weekday, self.STEPS_PER_DAY)
        valid_length = FlowModelPL._convert_offset_month_length(
            month_length, 28, self.STEPS_PER_DAY
        )
        mask = FlowModelPL._create_loss_mask(
            torch.tensor([valid_length]), self.FULL_LENGTH
        )
        return start_pos, mask, valid_length

    def test_feb_28_padding_zeroed(self):
        start_pos, mask, valid_length = self._pipeline(2023, 1)  # Feb 2023
        assert valid_length == 2688
        # Simulate profile * mask
        profile = torch.randn(1, self.FULL_LENGTH)
        masked = profile * mask
        assert torch.all(masked[0, 2688:] == 0.0)
        assert not torch.all(masked[0, :2688] == 0.0)  # valid region not zeroed

    def test_jan_31_no_padding(self):
        _, mask, valid_length = self._pipeline(2024, 0)  # Jan 2024
        assert valid_length == 2976
        assert torch.all(mask == 1.0)

    def test_mixed_batch(self):
        """Batch of Feb 2023 (28d) and Jan 2024 (31d)."""
        _, days_feb = calendar.monthrange(2023, 2)
        _, days_jan = calendar.monthrange(2024, 1)
        ml = torch.tensor([days_feb - 28, days_jan - 28])
        valid = FlowModelPL._convert_offset_month_length(ml, 28, self.STEPS_PER_DAY)
        mask = FlowModelPL._create_loss_mask(valid, self.FULL_LENGTH)

        profile = torch.randn(2, self.FULL_LENGTH)
        masked = profile * mask

        # Feb: padding zeroed
        assert torch.all(masked[0, 2688:] == 0.0)
        # Jan: no padding
        assert torch.equal(masked[1], profile[1])
