import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_all_candidates import nondominated


def test_tradeoff_does_not_create_an_unqualified_winner():
    vectors = {"base": (20, 36, 19, -0.04, -3.0),
               "chirality": (25, 36, 23, -0.05, -3.0),
               "geometry": (20, 36, 19, -0.03, -2.5),
               "worse": (19, 35, 18, -0.05, -3.5)}
    assert set(nondominated(vectors)) == {"chirality", "geometry"}


def test_tied_candidates_are_both_retained():
    assert nondominated({"a": (1, 2), "b": (1, 2)}) == ["a", "b"]
