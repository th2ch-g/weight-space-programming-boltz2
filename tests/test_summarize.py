import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize import median_difference_interval


def test_difference_of_medians_is_not_median_paired_change():
    base = np.array([1.0, 10.0, 11.0])
    edited = np.array([9.0, 2.0, 12.0])
    groups = [np.array([0, 1, 2])]
    draws = np.zeros((10, 1), dtype=int)
    assert median_difference_interval(base, edited, groups, draws) == [-1.0, -1.0]
    assert np.median(edited - base) == 1.0


def test_cluster_draws_repeat_every_system_in_the_selected_group():
    base = np.array([0.0, 0.0, 100.0])
    edited = np.array([3.0, 3.0, 99.0])
    groups = [np.array([0, 1]), np.array([2])]
    draws = np.array([[0, 1], [0, 0], [1, 1]])
    expected = np.percentile([3.0, 3.0, -1.0], [2.5, 97.5])
    assert np.allclose(median_difference_interval(base, edited, groups, draws), expected)
