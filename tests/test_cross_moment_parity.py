import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_cross_moment_parity import cross_moment


def test_cross_moment_equals_weighted_oriented_tetrahedron_products():
    rng = np.random.default_rng(51)
    for size in (4, 5):
        reference = rng.normal(size=(size, 3))
        predicted = rng.normal(size=(size, 3))
        ref_homogeneous = np.column_stack((reference, np.ones(size)))
        pred_homogeneous = np.column_stack((predicted, np.ones(size)))
        for weight in (1.0, 1 / 256):
            expected = sum(
                np.linalg.det(ref_homogeneous[list(indices)])
                * np.linalg.det(pred_homogeneous[list(indices)])
                * (weight if 0 in indices else 1.0)
                for indices in itertools.combinations(range(size), 4)
            )
            np.testing.assert_allclose(cross_moment(reference, predicted, weight), expected, rtol=1e-10)


def test_finite_center_weight_is_not_a_universal_four_neighbor_classifier():
    neighbors = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    reference = np.vstack(([0.2, 0, 0], neighbors))
    predicted = np.vstack(([-100000, 0, 0], neighbors))
    assert cross_moment(reference, predicted, 0.0) > 0
    assert cross_moment(reference, predicted, 1 / 256) < 0
