import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from trace_reservation import channel_statistics


def test_large_constant_channel_controls_layer_norm_without_between_token_variance():
    values = torch.tensor([[1000., 1., -1., 0.], [1000., -1., 1., 0.]])
    original = values.clone()
    report = channel_statistics(values, [0])
    zeroed = values.clone()
    zeroed[:, 0] = 0
    expected = (zeroed.var(-1, unbiased=False) / values.var(-1, unbiased=False)).sqrt().mean()
    assert report["selected_std"] == [0.0]
    assert report["mean_std_ratio_if_selected_zeroed"] == pytest.approx(float(expected))
    assert report["mean_centered_variance_fraction_in_selected"] > 0.7
    assert torch.equal(original, values)
